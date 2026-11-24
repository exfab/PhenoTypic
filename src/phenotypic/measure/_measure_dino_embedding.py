"""``MeasureDinoEmbedding``: DINO feature vectors of colonies or grid cells (#281 spec E3)."""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Final, Optional, TypeAlias

import numpy as np
from pydantic import Field, field_validator, model_validator

from phenotypic.abc_ import BackboneInputs, MeasureEmbedding, PreparedRegions, RegionEncoder
from phenotypic.abc_._measure_embedding import GridCellOutsideImageError
from phenotypic.schema import DINO_EMBEDDING, MeasurementInfo
from phenotypic.sdk_._embedding_tables import EMBEDDING_COLUMN
from phenotypic.sdk_.typing_ import (
    COMPUTE_DTYPE_DEFAULT,
    ComputeDType,
    DinoVersion,
    EmbeddingDevice,
    EmbeddingDinoSize,
    EmbeddingDType,
    EmbeddingPooling,
    TuneSpec,
)

if TYPE_CHECKING:
    from phenotypic._core._image import Image
    from phenotypic.detect.nn._helper._batch_encoder import BackboneKey
    from phenotypic.abc_._requirements import OperationRequirements

    # The pydantic mypy plugin resolves the inherited runtime OperationField
    # annotation (``grid``) in each concrete model's module, as in
    # ``_measure_symzones.py``. Runtime validation stays on MeasureEmbedding.
    OperationField: TypeAlias = Any

__all__ = ["GridCellOutsideImageError", "MeasureDinoEmbedding"]

#: ViT patch size of each DINO version's checkpoints: DINOv2 is ViT-*/14, the
#: DINOv3 LVD-1689M checkpoints are ViT-*/16. The model token spells it
#: before any weights load; the loaded backbone is checked against it.
_PATCH_SIZE: Final[dict[int, int]] = {2: 14, 3: 16}

#: ``size`` -> the ViT letter of the model token (``vits``, ``vitb``, ``vitl``,
#: ``vitg``).
_SIZE_LETTER: Final[dict[str, str]] = {"small": "s", "base": "b", "large": "l", "giant": "g"}

#: The sizes each DINO version has: DINOv3 has no ViT-g (#282 U11).
_VERSION_SIZES: Final[dict[int, tuple[str, ...]]] = {
    2: ("small", "base", "large", "giant"),
    3: ("small", "base", "large"),
}


def load_dino_backbone(
    dino_version: DinoVersion,
    dino_size: EmbeddingDinoSize,
    device: str,
    *,
    revision: str | None = None,
    dtype: ComputeDType = "float32",
) -> tuple[Any, Any]:
    """Load a frozen DINO ``(model, processor)``; see ``_dino_support.load_dino_backbone``.

    A thin lazy delegate, so importing ``phenotypic.measure`` does not import
    ``phenotypic.detect`` (whose package import pulls in matplotlib and the
    detectors). Tests patch this name to load a stub instead.
    """
    from phenotypic.detect.nn._helper import _dino_support

    return _dino_support.load_dino_backbone(
        dino_version, dino_size, device, revision=revision, dtype=dtype
    )


def load_dino_processor(
    dino_version: DinoVersion,
    dino_size: EmbeddingDinoSize,
    *,
    revision: str | None = None,
) -> Any:
    """Load a DINO backbone's processor only; see ``_dino_support.load_dino_processor``.

    A thin lazy delegate, as :func:`load_dino_backbone` is, which tests patch
    alongside it: what :meth:`MeasureDinoEmbedding.prepare_regions`
    normalises ``masked_dense`` tiles with when it is given no processor.
    """
    from phenotypic.detect.nn._helper import _dino_support

    return _dino_support.load_dino_processor(dino_version, dino_size, revision=revision)


@functools.lru_cache(maxsize=2)
def _cached_backbone(
    version: int, size: str, revision: str | None, device: str, compute_dtype: str
) -> tuple[Any, Any]:
    """One ``(model, processor)`` per process for each backbone identity.

    ``(version, size)`` names the Hugging Face id one-to-one
    (``hf_dino_id``), so this is the ``(hf_id, revision, device)`` cache of
    spec E3, keyed on the ``compute_dtype`` the weights load in too (#282).
    Two entries: one pipeline rarely holds more than one embedder, and a
    ViT-L is over a gigabyte.
    """
    return load_dino_backbone(
        version, size, device, revision=revision, dtype=compute_dtype  # type: ignore[arg-type]
    )


class MeasureDinoEmbedding(MeasureEmbedding):
    """Embed each colony, or each cell of a fixed grid, with a frozen DINO ViT.

    Each detected colony (``regions="objects"``) or each cell of a per-rig
    grid (``regions="grid"``) becomes one feature vector from a pretrained
    DINOv2 or DINOv3 backbone. The vectors capture a colony's appearance --
    texture, colour, edge, morphology -- without a hand-picked feature, so
    they suit clustering, nearest-neighbour search and outlier screens across
    a screen, and comparisons of a grid cell's growth across time points. The
    model is never trained here: the same weights give the same vector.

    The output is not a set of measurement columns. Each image gets one
    embedding table, ``Object_Label`` (or the ``Grid_*`` keys) plus an
    ``embedding`` vector per row, in its store's run folder, and the run
    gets ``deliverables/embeddings/<model>/<layer>/<region>.parquet``: the
    layer ``rgb`` or ``gray``, the region ``objects`` or ``grid``, or
    ``objects-<name>`` / ``grid-<name>`` when named;
    :class:`~phenotypic.schema.DINO_EMBEDDING` documents the vector.
    :meth:`ImagePipeline.embed` runs it; :meth:`ImagePipeline.measure` does
    not. It runs on the CPU unless CUDA is available.

    **Pooling.** ``crop_cls`` and ``crop_mean`` crop each region, resize the
    crop to ``crop_px`` square and take the CLS token or the mean patch
    token. A colony's crop is its bounding box padded by ``crop_pad_frac``
    and squared to its longer side about its centre, so an elongated or
    filamentous colony keeps its shape; where the square leaves the image it
    is zero-filled rather than stretched. A grid cell is cropped exactly. ``masked_dense`` runs the backbone once over the whole plate at
    native resolution, in ``crop_px`` tiles, and averages the patch tokens
    under each region's mask (the colony's own pixels, or the cell): finer
    context, no resizing, and about ``H/p * W/p * dim`` float32 of memory for
    the plate's feature grid. ``crop_pad_frac``, ``mask_background`` and
    ``batch_size`` do not apply to it.

    **Identity.** The weights are named by the model token
    ``dinov{version}-vit{s|b|l|g}{patch}`` (``dinov2-vitb14``), the folder its
    tables go in, and by ``revision``. Each table's entry records the
    Hugging Face commit actually loaded, pinned or not. The backbone loads
    once per process, and each process holds its own copy: a local run with
    ``--njobs N`` holds N, about 90 MB each for ViT-S, 350 MB for ViT-B and
    1.2 GB for ViT-L and 4.4 GB for ViT-g in float32 (half that in
    ``compute_dtype="bfloat16"``), on top of each worker's images. DINOv3
    weights are gated: the run refuses unless
    ``PHENOTYPIC_ACCEPT_MODEL_LICENSE`` lists ``dinov3``, and never prompts.

    Args:
        size: Backbone size, ``"small"`` (ViT-S, 384-d vectors), ``"base"``
            (ViT-B, 768-d), ``"large"`` (ViT-L, 1024-d) or ``"giant"``
            (ViT-g, 1536-d; DINOv2 only, since DINOv3 has no ViT-g).
        version: DINO version, ``2`` (DINOv2, patch 14, ungated) or ``3``
            (DINOv3, patch 16, gated by the DINOv3 License).
        revision: The weights' Hugging Face revision, a commit hash, branch
            or tag. Pin it to a commit for reproducible vectors. ``None``
            loads the hub's latest in each process, so two runs, or two
            workers of one run, may embed with different weights while the
            pipeline file stays the same; the run preflight warns
            (``PF-EMBED-REVISION-UNPINNED``), and finalization tags each
            deliverable row with the ``descriptor_index`` of the weights it
            came from. A pin lives in the pipeline file, so changing it
            changes the pipeline's hash.
        regions: ``"objects"`` embeds each detected colony; ``"grid"``
            embeds each cell of ``grid`` and needs no detection.
        grid: A :class:`~phenotypic.grid.ManualGridFinder` with the rig's
            cell edges in pixels of the image the measurer sees. Required
            with ``regions="grid"``, refused otherwise. A cell outside the
            image fails that image rather than being clipped.
        name: Names this embedder's table, so one pipeline can hold several
            embedders of one model and region -- a rig's well lattice and
            one whole-agar cell, say, or two poolings of the colonies. Its
            region is then ``grid-<name>`` (``dinov2-vitb14/rgb/grid-wells``) or
            ``objects-<name>`` instead of ``grid`` or ``objects``. A lowercase
            slug of 1 to 32 characters: ``a-z``, ``0-9`` and ``_``, starting
            with a letter or digit. ``None`` keeps the plain region. Two
            embedders of one model, layer and region need different names;
            the backbone still loads once for both.
        layer: The pixels embedded. ``"rgb"`` (the default) is the colour
            layer; ``"gray"`` is the gray layer, scaled to uint8 as an RGB
            crop is and stacked to three identical channels, then normalised
            as usual. Every pooling, ``mask_background`` and grid cell works
            on either. It is the table's ``<layer>`` folder, so an rgb and a
            gray copy of one embedder coexist, even with one ``name``; its
            ``params_sha256`` differs too.
        pooling: How a region becomes one vector: ``"crop_cls"``,
            ``"crop_mean"`` or ``"masked_dense"`` (above).
        crop_px: Side of the square crop fed to the backbone, and the tile
            side for ``masked_dense``. Must be a multiple of the backbone's
            patch size (14 for DINOv2, 16 for DINOv3), checked when the
            backbone loads. Tuned over 112, 224, 336 and 448, the common
            multiples of 14 and 16 from 112 to 448: at or above both models'
            smallest training crop and below DINOv2's 518 training
            resolution; larger crops only upsample colony-sized boxes at
            quadratic attention cost.
        crop_pad_frac: Context around a colony, **per side**: its bounding
            box grows by this fraction of its height and width on each side,
            so the crop is ``1 + 2 * crop_pad_frac`` times the box (1.5x at
            the default 0.25). Tuned over 0.0-0.5; beyond that a crop on a
            dense plate takes in the neighbouring colonies. Objects only.
        mask_background: Zero the pixels of a colony's crop outside the
            colony, so only the colony is embedded. Objects only.
        dtype: The stored vector type. Only ``"float16"``.
        compute_dtype: The type the backbone runs in: ``"float32"`` (the
            default), ``"bfloat16"`` or ``"float16"``. The weights load in
            it and every forward pass runs in it; vectors are pooled in
            float32 and stored as ``dtype``. A reduced type halves the
            backbone's memory, which suits ``"giant"`` on a GPU, but moves
            the vectors far more than a change of device does, so it is part
            of the table's descriptor (``compute_dtype``): tables computed in
            different types are never grouped together. ``"float16"`` is
            accepted but **unmeasured** (#282 P2 T4 measured float32 and
            bfloat16 only); ViT-g is known to overflow float16 activations,
            which gives non-finite vectors and so a failure for every image.
        batch_size: Crops per forward pass. Changes speed and memory. On the
            CPU the vectors do not depend on it; on a GPU, batched kernels may
            round the last float16 bits differently.
        device: ``"auto"`` (CUDA when available, else the CPU), ``"cpu"`` or
            ``"cuda"``. Identical runs give byte-identical embedding
            deliverables only on the same device and the same BLAS path, but
            the vectors of different devices are interchangeable: a CPU and
            a GPU embedding of one colony differ by about 1e-6 in cosine
            distance, far below what hours of growth change, and two GPU
            models by about the last float16 bit. So the device is
            provenance, not part of the descriptor: each table's entry
            records the device that ran (``model.device``, ``"cuda"`` or
            ``"cpu"``) and on CUDA the GPU's name (``model.device_name``),
            and a run mixing devices still publishes one
            ``descriptor_index``. The entry's ``compute`` block records the
            same device beside the GPU's compute capability, the torch and
            CUDA versions, the inputs per forward pass and the stage that
            ran it; it is provenance too.

    Examples:
        Construction needs no weights; the backbone loads on the first
        :meth:`measure`.

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.grid import ManualGridFinder
        >>> from phenotypic.measure import MeasureDinoEmbedding
        >>> colonies = MeasureDinoEmbedding(size="small")
        >>> colonies.model_token(), colonies.payload_key()
        ('dinov2-vits14', 'dinov2-vits14/rgb/objects')
        >>> whole_plate = MeasureDinoEmbedding(
        ...     regions="grid", grid=ManualGridFinder(row_edges=[0, 600], col_edges=[0, 800])
        ... )
        >>> whole_plate.payload_key(), whole_plate.requires_objects
        ('dinov2-vitb14/rgb/grid', False)

        Two grids of one model, each in its own table: a rig's 2 x 3 well
        lattice and one cell over the whole agar it covers.

        >>> wells = MeasureDinoEmbedding(
        ...     regions="grid", name="wells",
        ...     grid=ManualGridFinder(row_edges=[0, 300, 600], col_edges=[0, 260, 530, 800]),
        ... )
        >>> agar = MeasureDinoEmbedding(
        ...     regions="grid", name="agar",
        ...     grid=ManualGridFinder(row_edges=[0, 600], col_edges=[0, 800]),
        ... )
        >>> wells.payload_key(), agar.payload_key()
        ('dinov2-vitb14/rgb/grid-wells', 'dinov2-vitb14/rgb/grid-agar')
        >>> rig = ImagePipeline(meas={"wells": wells, "agar": agar})

        The same wells on the gray layer sit beside them, in ``gray/``:

        >>> wells_gray = MeasureDinoEmbedding(
        ...     regions="grid", name="wells", layer="gray", grid=wells.grid,
        ... )
        >>> wells_gray.payload_key()
        'dinov2-vitb14/gray/grid-wells'
        >>> rig = ImagePipeline(meas={"wells": wells, "wells_gray": wells_gray, "agar": agar})
        >>> pipe = ImagePipeline(ops=[OtsuDetector()], meas={"dino": colonies})

        With the weights available, ``pipe.embed(image)`` returns one
        payload per embedder, and ``colonies.measure(image)`` the table's frame:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> frame = colonies.measure(OtsuDetector().apply(load_synth_yeast_plate()))  # doctest: +SKIP
        >>> frame.columns.tolist(), frame["embedding"][0].shape  # doctest: +SKIP
        (['Object_Label', 'embedding'], (384,))
    """

    _measurement_infoclass: ClassVar = DINO_EMBEDDING
    _requires_rgb_input: ClassVar[bool] = True
    _leading_fields: ClassVar[tuple[str, ...]] = ("size", "version", "revision")

    size: Annotated[EmbeddingDinoSize, TuneSpec(tunable=False)] = "base"
    version: Annotated[DinoVersion, TuneSpec(tunable=False)] = 2
    revision: Annotated[Optional[str], TuneSpec(tunable=False)] = None
    pooling: EmbeddingPooling = "crop_cls"
    crop_px: Annotated[int, TuneSpec(categories=(112, 224, 336, 448))] = Field(224, gt=0)
    crop_pad_frac: Annotated[float, TuneSpec(0.0, 0.5)] = Field(0.25, ge=0.0)
    mask_background: bool = False
    dtype: Annotated[EmbeddingDType, TuneSpec(tunable=False)] = "float16"
    compute_dtype: Annotated[ComputeDType, TuneSpec(tunable=False)] = COMPUTE_DTYPE_DEFAULT
    batch_size: Annotated[int, TuneSpec(tunable=False)] = Field(64, ge=1)
    device: Annotated[EmbeddingDevice, TuneSpec(tunable=False)] = "auto"

    @field_validator("revision")
    @classmethod
    def _revision_is_not_blank(cls, value: str | None) -> str | None:
        """A pinned revision names something; ``""`` would silently load the latest."""
        if value is not None and not value.strip():
            raise ValueError("revision must be None or a non-empty Hugging Face revision")
        return value

    @model_validator(mode="after")
    def _size_matches_version(self) -> "MeasureDinoEmbedding":
        """A size the version has: ``giant`` is DINOv2 only (#282 U11)."""
        sizes = _VERSION_SIZES[int(self.version)]
        if self.size not in sizes:
            raise ValueError(
                f"size={self.size!r} needs version=2: DINOv{int(self.version)} has "
                f"no ViT-g. version={int(self.version)} has sizes "
                f"{', '.join(repr(size) for size in sizes)}"
            )
        return self

    @model_validator(mode="after")
    def _crop_options_are_objects_only(self) -> "MeasureDinoEmbedding":
        """``crop_pad_frac`` and ``mask_background`` are refused for a grid (spec E3).

        A grid cell is cropped exactly, so a non-default value would be
        silently ignored. The default ``crop_pad_frac`` is accepted, so a
        grid embedder's pipeline file, which spells every field, round-trips.
        """
        if self.regions == "grid":
            default_pad = type(self).model_fields["crop_pad_frac"].default
            if self.mask_background:
                raise ValueError(
                    "mask_background is for objects only: a grid cell has no "
                    "colony mask. Drop it, or set regions='objects'"
                )
            if self.crop_pad_frac != default_pad:
                raise ValueError(
                    f"crop_pad_frac is for objects only: a grid cell is cropped "
                    f"exactly. Leave it at {default_pad}, or set regions='objects'"
                )
        return self

    @classmethod
    def output_header(cls, member: MeasurementInfo, on: str | None = None) -> str:
        """The column :meth:`measure` writes for *member*: ``embedding``.

        An embedding table's vector column is named ``embedding`` in every
        table (spec D1), not ``DinoEmbedding_Embedding``, so that is the
        header the Measurements reference documents.

        Args:
            member: ``DINO_EMBEDDING.EMBEDDING``.
            on: Unused; accepted for the shared ``output_header`` signature.

        Returns:
            ``"embedding"``.
        """
        if member is DINO_EMBEDDING.EMBEDDING:
            return EMBEDDING_COLUMN
        return super().output_header(member, on)

    # ------------------------------------------------------------------ #
    # Model identity
    # ------------------------------------------------------------------ #

    def model_token(self) -> str:
        """``dinov{version}-vit{s|b|l|g}{patch}``, e.g. ``"dinov2-vitb14"``."""
        version = int(self.version)
        return f"dinov{version}-vit{_SIZE_LETTER[self.size]}{_PATCH_SIZE[version]}"

    def backbone_key(self) -> BackboneKey:
        """The model load this embedder needs, for Stage 4 to share it (#282 H10).

        Returns:
            ``BackboneKey(version, size, revision, compute_dtype)``
            (``phenotypic.detect.nn._helper._batch_encoder``): embedders with
            equal keys share one resident model.
        """
        from phenotypic.detect.nn._helper._batch_encoder import BackboneKey

        return BackboneKey(int(self.version), self.size, self.revision, self.compute_dtype)

    def entry_descriptor(self) -> dict:
        """The model, vector length, stored type and pooling of the table's entry.

        Loads the backbone (cached), because the vector length, the resolved
        revision and the device are properties of the weights actually
        loaded: :meth:`entry_descriptor_for` of that backbone.

        Returns:
            As :meth:`entry_descriptor_for`.
        """
        return self.entry_descriptor_for(self._backbone()[0])

    def entry_descriptor_for(
        self, model: Any, compute: Mapping[str, Any] | None = None
    ) -> dict:
        """:meth:`entry_descriptor` of an already loaded *model*: loads nothing (#282).

        A GPU stage that holds its own model describes the entry with it,
        rather than through :meth:`_backbone`, which would load a second one.

        Args:
            model: The loaded backbone the vectors came from.
            compute: The entry's ``compute`` block, from
                :func:`~phenotypic.detect.nn._helper._dino_support.compute_block`.
                ``None`` builds a nominal inline one: this embedder's
                resolved ``device``, ``stage="inline"``, and the
                ``batch_size`` cap per forward pass (one tile for
                ``masked_dense``, which runs a tile at a time). A payload's
                entry records the image's own passes instead
                (:meth:`_inline_entry_descriptor`, #282 review M10).

        Returns:
            The descriptor ``{model, dim, dtype, compute_dtype, pooling,
            compute}``, whose ``model`` is ``{family, hf_id, revision,
            patch_size, device, device_name}``. ``revision`` is the commit loaded
            (:func:`~phenotypic.detect.nn._helper._dino_support.resolved_revision`),
            and ``device`` (``"cuda"`` or ``"cpu"``) and ``device_name`` (the
            GPU's name on CUDA, ``None`` on the CPU) are copied from
            *compute*, their one producer (#282 O4). The device fields and
            ``compute`` are provenance: the run's deliverables group
            descriptors without them (#281, user decision 2026-11-19).
            ``compute_dtype`` is identity.
        """
        from phenotypic.detect.nn._helper import _dino_support

        if compute is None:
            physical_batch = 1 if self.pooling == "masked_dense" else self.batch_size
            compute = _dino_support.compute_block(
                device=_dino_support.resolve_backbone_device(self.device),
                stage="inline",
                physical_batch=physical_batch,
            )
        block = dict(compute)
        return {
            "model": {
                "family": f"dinov{int(self.version)}",
                "hf_id": _dino_support.hf_dino_id(self.version, self.size),
                "revision": _dino_support.resolved_revision(model, self.revision),
                "patch_size": _dino_support.backbone_patch_size(model),
                "device": block["device"],
                "device_name": block["device_name"],
            },
            "dim": int(model.config.hidden_size),
            "dtype": self.dtype,
            "compute_dtype": self.compute_dtype,
            "pooling": self.pooling,
            "compute": block,
        }

    def crop_geometry(self) -> tuple[int, int]:
        """``(crop_px, crop_px)``: the shape of each crop :meth:`prepare_regions` cuts."""
        return (self.crop_px, self.crop_px)

    def preflight_requirements(self) -> OperationRequirements:
        """Packages and weights this measurer loads (run preflight).

        torch and transformers (the ``foundation`` extra), plus
        huggingface_hub for gated DINOv3; RGB input, unless ``layer="gray"``;
        and the backbone's weights. See ``BaseOperation.preflight_requirements``.
        """
        import dataclasses

        from phenotypic.detect.nn._helper import _checkpoint_manager as ckpt

        requirements = super().preflight_requirements()
        modules: tuple[str, ...] = ("transformers", "torch")
        if int(self.version) == 3:
            modules += ("huggingface_hub",)
        return dataclasses.replace(
            requirements,
            rgb_input=self.layer == "rgb",
            modules=modules,
            extra="foundation",
            weights=(ckpt.dino_weight_requirement(self.version, self.size),),
        )

    # ------------------------------------------------------------------ #
    # The backbone
    # ------------------------------------------------------------------ #

    def _backbone(self) -> tuple[Any, Any, str]:
        """The cached ``(model, processor, device)``, after the license gate.

        Raises:
            RuntimeError: For DINOv3, if its license is not accepted
                (before anything loads).
            ValueError: If the loaded patch size is not the version's, or
                ``crop_px`` is not a multiple of it.
        """
        if int(self.version) == 3:
            from phenotypic.detect.nn._helper._checkpoint_manager import (
                Dinov3CheckpointManager,
            )

            # A batch worker has no terminal: refuse rather than prompt.
            Dinov3CheckpointManager.require_license(interactive=False)
        from phenotypic.detect.nn._helper._dino_support import resolve_backbone_device

        device = resolve_backbone_device(self.device)
        model, processor = _cached_backbone(
            int(self.version), self.size, self.revision, device, self.compute_dtype
        )
        self._check_geometry(model)
        return model, processor, device

    def _check_geometry(self, model: Any) -> None:
        from phenotypic.detect.nn._helper._dino_support import backbone_patch_size

        patch = backbone_patch_size(model)
        expected = _PATCH_SIZE[int(self.version)]
        if patch != expected:
            raise ValueError(
                f"the loaded DINOv{int(self.version)} backbone has patch size "
                f"{patch}, not {expected}; its model token "
                f"{self.model_token()!r} would misname it"
            )
        if self.crop_px % patch:
            raise ValueError(
                f"crop_px={self.crop_px} is not a multiple of the patch size "
                f"{patch}; use one of 112, 224, 336 or 448"
            )

    # ------------------------------------------------------------------ #
    # Embedding
    # ------------------------------------------------------------------ #

    def prepare_regions(
        self, image: Image, *, processor: Any = None, materialise: bool = True
    ) -> PreparedRegions:
        """Cut *image*'s regions for the backbone, on the CPU, loading no model (#282).

        ``crop_cls`` / ``crop_mean``: every region's crop, through
        :meth:`_crop` and :meth:`_square_colony_crop`, as one
        ``(N, crop_px, crop_px, 3)`` uint8 array (``"crops"``). That is every
        crop at once, about 0.9 GB at ``crop_px=448`` on a 1536-colony plate,
        which a DataLoader worker can hold. The inline path passes
        ``materialise=False`` instead (:meth:`_prepare_inline`, #282 I4): the
        crops are then cut one ``batch_size`` slice at a time as its encoder
        reads them, so host memory holds one batch of crops, as it did before
        #282; the bytes are the same ``_crop`` calls in the same order.
        Imports no torch.

        ``masked_dense``: the plate cut into ``crop_px`` tiles aligned to the
        patch grid, each normalised by the backbone's processor exactly as
        :func:`~phenotypic.detect.nn._helper._dino_support.extract_patch_features`
        does it (``"tiles"``), with the feature-grid shape, each tile's
        offset and each region's mask in ``pool_plan``. The HF processor may
        import torch.

        Args:
            image: The image to embed.
            processor: The backbone's processor, for ``masked_dense``.
                ``None`` loads one (:func:`load_dino_processor`), without the
                weights. Crop pooling reads none.
            materialise: ``False`` defers the crops: ``arrays[0]`` is then a
                :class:`_DeferredCrops`, which holds the image's pixels and is
                for the inline encoder in this process only, never for
                pickling to a worker. ``masked_dense`` tiles ignore it.

        Returns:
            The prepared regions, in table order.

        Raises:
            GridCellOutsideImageError: As :meth:`MeasureEmbedding.prepare_regions`.
            ValueError: For ``masked_dense``, if the image is smaller than
                one patch.
        """

        def processor_of() -> Any:
            if processor is not None:
                return processor
            return load_dino_processor(self.version, self.size, revision=self.revision)

        return self._prepare_regions(image, processor_of=processor_of, materialise=materialise)

    def _prepare_regions(
        self, image: Image, *, processor_of: Callable[[], Any], materialise: bool
    ) -> PreparedRegions:
        """:meth:`prepare_regions`, with the processor read from *processor_of* only when
        ``masked_dense`` cuts tiles (after the regions, so a grid error still comes first)."""
        keys, boxes, labels = self._regions(image)
        rgb = self._layer_pixels(image)
        needs_mask = self.mask_background or self.pooling == "masked_dense"
        objmap = np.asarray(image.objmap[:]) if labels is not None and needs_mask else None
        if self.pooling == "masked_dense":
            inputs, pool_plan = self._dense_tiles(rgb, boxes, labels, objmap, processor_of)
            return self._prepared(image, keys, inputs, pool_plan)

        region_labels: list[int | None] = (
            [None] * len(boxes) if labels is None else [int(v) for v in labels]
        )
        deferred = _DeferredCrops(self, rgb, boxes, region_labels, objmap)
        crops = deferred[:] if materialise else deferred
        inputs = BackboneInputs(kind="crops", arrays=(crops,), pooling=self.pooling)  # type: ignore[arg-type]
        return self._prepared(image, keys, inputs)

    def _prepare_inline(self, image: Image) -> PreparedRegions:
        """:meth:`prepare_regions` as the inline path needs it (#282 review I4, M11).

        The crops are deferred to the inline encoder, and ``masked_dense``
        tiles are normalised by the cached backbone's own processor, the one
        its weights came with, rather than a second ``from_pretrained``.
        """
        return self._prepare_regions(
            image, processor_of=lambda: self._backbone()[1], materialise=False
        )

    def _inline_entry_descriptor(self, prepared: PreparedRegions) -> dict:
        """The inline entry's descriptor, its ``compute`` block stating this image's passes (M10).

        ``physical_batch`` is the pass size the inline encoder actually ran,
        not the ``batch_size`` cap: a crop image of ``N`` regions runs
        ``N // batch_size`` passes of ``batch_size`` and, unpadded, one of
        ``N % batch_size``, and CPU float32 bits depend on that size (C5).
        ``masked_dense`` runs one tile a pass; an image with no regions runs
        none.
        """
        from phenotypic.detect.nn._helper import _dino_support

        if prepared.n_regions == 0:
            sizes: list[int] = []
        elif prepared.inputs.kind == "tiles":
            sizes = [1]
        else:
            sizes = [stop - start for start, stop in _batch_bounds(prepared.n_regions, self.batch_size)]
        compute = _dino_support.compute_block(
            device=_dino_support.resolve_backbone_device(self.device),
            stage="inline",
            physical_batch=sizes,
        )
        return self.entry_descriptor_for(self._backbone()[0], compute)

    def _dense_tiles(
        self,
        rgb: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray | None,
        objmap: np.ndarray | None,
        processor_of: Callable[[], Any],
    ) -> tuple[BackboneInputs, dict[str, Any] | None]:
        """``masked_dense``'s tiles and the plan to pool their patch grids.

        The plate is cut into ``crop_px`` tiles aligned to the patch grid, so
        their ``(h // p, w // p)`` grids abut exactly in one
        ``(H // p, W // p, dim)`` feature grid and no patch is computed
        twice. The bottom and right ``H % p`` / ``W % p`` pixels are not
        covered, as in every DINO dense path here. An image with no regions
        gets no tiles: it never reaches the backbone.
        """
        if len(boxes) == 0:
            return BackboneInputs(kind="tiles", arrays=(), pooling="patches"), None
        from phenotypic.detect.nn._helper._dino_support import NATIVE_PROCESSOR_KWARGS

        patch = _PATCH_SIZE[int(self.version)]
        height, width = rgb.shape[:2]
        grid_h, grid_w = height // patch, width // patch
        if grid_h == 0 or grid_w == 0:
            raise ValueError(
                f"the image is {height} x {width}, smaller than one {patch}-px patch"
            )
        processor = processor_of()
        tiles: list[np.ndarray] = []
        offsets: list[tuple[int, int]] = []
        tile = self.crop_px
        for y0 in range(0, grid_h * patch, tile):
            for x0 in range(0, grid_w * patch, tile):
                y1, x1 = min(y0 + tile, grid_h * patch), min(x0 + tile, grid_w * patch)
                pixels = processor(
                    images=_as_uint8_rgb(rgb[y0:y1, x0:x1]),
                    return_tensors="np",
                    **NATIVE_PROCESSOR_KWARGS,
                )["pixel_values"]
                tiles.append(np.asarray(pixels[0]))
                offsets.append((y0 // patch, x0 // patch))

        masks: list[np.ndarray] = []
        for i, box in enumerate(boxes):
            r0, c0, r1, c1 = (int(v) for v in box)
            if labels is None:
                masks.append(np.ones((r1 - r0, c1 - c0), dtype=bool))
            else:
                assert objmap is not None
                masks.append(objmap[r0:r1, c0:c1] == int(labels[i]))
        pool_plan = {
            "patch": patch,
            "grid_hw": (grid_h, grid_w),
            "offsets": tuple(offsets),
            "origins": tuple((int(box[0]), int(box[1])) for box in boxes),
            "masks": tuple(masks),
        }
        return BackboneInputs(kind="tiles", arrays=tuple(tiles), pooling="patches"), pool_plan

    def inline_encoder(self, image: Image) -> RegionEncoder:
        """The cached backbone, on this embedder's device (after the license gate).

        Raises:
            RuntimeError: As :meth:`_backbone`.
            ValueError: As :meth:`_backbone`.
        """
        model, processor, device = self._backbone()
        return _InlineDinoEncoder(model, processor, device, self.batch_size)

    def vectors_from(
        self, prepared: PreparedRegions, outputs: list[np.ndarray]
    ) -> np.ndarray:
        """The ``(N, D)`` float32 vectors, region order, from the encoder's *outputs*.

        Crop pooling: the encoder's one ``(N, D)`` array. ``masked_dense``:
        each tile's ``(h // p, w // p, D)`` grid is placed in the plate's
        feature grid, and each region's vector is the mean feature under its
        mask (:func:`_pool_under_mask`).
        """
        if prepared.inputs.pooling != "patches":
            return super().vectors_from(prepared, outputs)
        plan = prepared.pool_plan
        assert plan is not None
        if len(outputs) != len(plan["offsets"]):
            raise ValueError(
                f"the encoder returned {len(outputs)} patch grids for "
                f"{len(plan['offsets'])} tiles"
            )
        grid_h, grid_w = plan["grid_hw"]
        features = np.zeros((grid_h, grid_w, int(outputs[0].shape[-1])), dtype=np.float32)
        for (gy, gx), dense in zip(plan["offsets"], outputs):
            features[gy : gy + dense.shape[0], gx : gx + dense.shape[1]] = dense
        vectors = np.empty((len(plan["masks"]), features.shape[-1]), dtype=np.float32)
        for i, (origin, region) in enumerate(zip(plan["origins"], plan["masks"])):
            vectors[i] = _pool_under_mask(features, region, origin, plan["patch"])
        return vectors

    def _crop(
        self,
        rgb: np.ndarray,
        box: np.ndarray,
        label: int | None,
        objmap: np.ndarray | None,
    ) -> np.ndarray:
        """One region as a ``(crop_px, crop_px, 3)`` uint8 crop.

        A colony's box (``label`` given) is padded by ``crop_pad_frac`` of its
        height and width on each side, squared to its longer side about its
        centre, and clipped to the image; what the clip removed is zero
        padding, so the colony is never stretched. With ``mask_background``
        the pixels outside the colony are zeroed. A grid cell
        (``label is None``) is its rectangle exactly. Either is then resized
        to ``crop_px`` square.
        """
        from skimage.transform import resize

        r0, c0, r1, c1 = (int(v) for v in box)
        if label is None:
            crop = _as_uint8_rgb(rgb[r0:r1, c0:c1])
        else:
            crop = self._square_colony_crop(rgb, (r0, c0, r1, c1), label, objmap)
        resized = resize(
            crop, (self.crop_px, self.crop_px), order=1, preserve_range=True
        )
        return np.clip(np.rint(resized), 0, 255).astype(np.uint8)

    def _square_colony_crop(
        self,
        rgb: np.ndarray,
        box: tuple[int, int, int, int],
        label: int,
        objmap: np.ndarray | None,
    ) -> np.ndarray:
        """A colony's padded, squared crop at native resolution, zero outside the image.

        Args:
            rgb: The embedder's ``(H, W, 3)`` layer pixels (:meth:`_layer_pixels`).
            box: The colony's end-exclusive ``(r0, c0, r1, c1)`` bounding box.
            label: Its ``Object_Label``.
            objmap: The objmap, needed with ``mask_background``.

        Returns:
            ``(side, side, 3)`` uint8, ``side`` the padded box's longer side.
        """
        r0, c0, r1, c1 = box
        height, width = rgb.shape[:2]
        pad_r = int(round(self.crop_pad_frac * (r1 - r0)))
        pad_c = int(round(self.crop_pad_frac * (c1 - c0)))
        r0, r1, c0, c1 = r0 - pad_r, r1 + pad_r, c0 - pad_c, c1 + pad_c
        side = max(r1 - r0, c1 - c0)
        r0 -= (side - (r1 - r0)) // 2
        c0 -= (side - (c1 - c0)) // 2
        r1, c1 = r0 + side, c0 + side

        # The part of the square inside the image, and where it sits in the square.
        ir0, ir1 = max(0, r0), min(height, r1)
        ic0, ic1 = max(0, c0), min(width, c1)
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        inside = _as_uint8_rgb(rgb[ir0:ir1, ic0:ic1])
        if self.mask_background:
            assert objmap is not None
            colony = objmap[ir0:ir1, ic0:ic1] == label
            inside = np.where(colony[..., None], inside, np.uint8(0))
        canvas[ir0 - r0 : ir1 - r0, ic0 - c0 : ic1 - c0] = inside
        return canvas


class _DeferredCrops:
    """An image's ``(N, crop_px, crop_px, 3)`` uint8 crops, cut when a slice is read (#282 I4).

    What the inline path's ``"crops"`` input holds: slicing ``[a:b]`` cuts
    crops ``a..b-1`` through :meth:`MeasureDinoEmbedding._crop`, in order,
    into a new array, so :class:`_InlineDinoEncoder` holds one ``batch_size``
    slice of crops at a time. ``[:]`` (and ``numpy.asarray``) cuts them all,
    which is what :meth:`MeasureDinoEmbedding.prepare_regions` returns by
    default. It holds the image's pixels: not for pickling to a worker.
    """

    dtype = np.dtype(np.uint8)
    ndim = 4

    def __init__(
        self,
        embedder: MeasureDinoEmbedding,
        rgb: np.ndarray,
        boxes: np.ndarray,
        labels: list[int | None],
        objmap: np.ndarray | None,
    ) -> None:
        self._embedder, self._rgb, self._boxes = embedder, rgb, boxes
        self._labels, self._objmap = labels, objmap
        px = int(embedder.crop_px)
        self.shape = (len(boxes), px, px, 3)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: slice) -> np.ndarray:
        if not isinstance(index, slice):
            raise TypeError(f"deferred crops are read by slice, got {index!r}")
        rows = range(len(self))[index]
        crops = np.empty((len(rows), *self.shape[1:]), dtype=np.uint8)
        for j, i in enumerate(rows):
            crops[j] = self._embedder._crop(self._rgb, self._boxes[i], self._labels[i], self._objmap)
        return crops

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        crops = self[:]
        return crops if dtype is None else crops.astype(dtype, copy=False)


class _InlineDinoEncoder:
    """The inline :class:`~phenotypic.abc_.RegionEncoder` of a loaded DINO backbone.

    ``"crops"``: :func:`~phenotypic.detect.nn._helper._dino_support.embed_crops`,
    called once per ``batch_size`` slice of the crops, so the device holds one
    batch at a time, and so does the host when the crops are deferred
    (:class:`_DeferredCrops`). ``"tiles"``: the backbone over each tile's processed
    pixels (:func:`~phenotypic.detect.nn._helper._dino_support.patch_features_from_pixels`).

    Args:
        model: The loaded backbone, on *device*.
        processor: Its processor; ``embed_crops`` reads its normalisation.
        device: The resolved device string.
        batch_size: Crops per forward pass.
    """

    def __init__(self, model: Any, processor: Any, device: str, batch_size: int) -> None:
        self.model, self.processor = model, processor
        self.device, self.batch_size = device, batch_size

    def run(self, inputs: BackboneInputs) -> list[np.ndarray]:
        from phenotypic.detect.nn._helper import _dino_support

        if inputs.kind == "crops":
            (crops,) = inputs.arrays
            vectors = [
                _dino_support.embed_crops(
                    self.model,
                    self.processor,
                    crops[start:stop],
                    pooling=inputs.pooling,  # type: ignore[arg-type]
                    device=self.device,
                    batch_size=self.batch_size,
                )
                for start, stop in _batch_bounds(len(crops), self.batch_size)
            ]
            if not vectors:
                hidden = int(self.model.config.hidden_size)
                return [np.zeros((0, hidden), dtype=np.float32)]
            return [np.concatenate(vectors, axis=0)]
        if inputs.kind == "tiles":
            return [
                _dino_support.patch_features_from_pixels(self.model, tile, device=self.device)
                for tile in inputs.arrays
            ]
        raise ValueError(f"a DINO backbone does not run {inputs.kind!r} inputs")


def _batch_bounds(n: int, batch_size: int) -> list[tuple[int, int]]:
    """The inline encoder's passes over *n* crops: ``(start, stop)``, ``batch_size`` each, the last
    shorter. Its one definition, read by the encoder and by the entry's ``compute`` block (M10)."""
    return [(start, min(start + batch_size, n)) for start in range(0, n, batch_size)]


def _as_uint8_rgb(array: np.ndarray) -> np.ndarray:
    """An RGB crop as ``uint8``, scaled by its dtype's range, as GPU detectors do.

    Integer crops map ``0..iinfo.max`` to ``0..255``; float crops map
    ``0..1``. A ``uint8`` crop is returned as is.
    """
    from phenotypic.abc_._gpu_detector import _scale_to_uint8

    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    return _scale_to_uint8(array, "dtype_range")


def _pool_under_mask(
    features: np.ndarray, region: np.ndarray, origin: tuple[int, int], patch: int
) -> np.ndarray:
    """Mean patch feature under one region's mask.

    The region's patch-aligned window is cut from the plate's feature grid,
    and the region's mask is downsampled onto it with ``resize_mask_to_grid``
    (nearest, so a patch counts when its centre pixel is in the region).
    A region too small for any patch centre falls back to every patch it
    touches, and a region in the uncovered bottom/right margin to its
    nearest covered patches, so a small colony never gets a zero vector.

    Args:
        features: ``(Hp, Wp, D)`` plate patch features.
        region: ``(h, w)`` boolean mask of the region's box.
        origin: ``(r0, c0)`` of the box in the image.
        patch: The backbone's patch size.

    Returns:
        ``(D,)`` float32.
    """
    from phenotypic.detect.nn._helper._dino_support import resize_mask_to_grid

    grid_h, grid_w = features.shape[:2]
    r0, c0 = origin
    h, w = region.shape
    pr0, pc0 = min(r0 // patch, grid_h - 1), min(c0 // patch, grid_w - 1)
    pr1 = min(max(-(-(r0 + h) // patch), pr0 + 1), grid_h)
    pc1 = min(max(-(-(c0 + w) // patch), pc0 + 1), grid_w)
    rows, cols = pr1 - pr0, pc1 - pc0

    window = np.zeros((rows * patch, cols * patch), dtype=bool)
    oy, ox = r0 - pr0 * patch, c0 - pc0 * patch
    ey, ex = min(h, window.shape[0] - oy), min(w, window.shape[1] - ox)
    if ey > 0 and ex > 0:
        window[oy : oy + ey, ox : ox + ex] = region[:ey, :ex]

    selected = resize_mask_to_grid(window, (rows, cols), patch)
    if not selected.any():
        selected = np.asarray(window.reshape(rows, patch, cols, patch).any(axis=(1, 3)))
    if not selected.any():
        selected = np.ones((rows, cols), dtype=bool)
    return features[pr0:pr1, pc0:pc1][selected].mean(axis=0)
