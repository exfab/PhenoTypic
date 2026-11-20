"""The vector column of a DINO embedding table."""

from .._base._measurement_info import Entry
from .._base._tiers import DiscriminativeFeature
from .._base._units import UNITS


class DINO_EMBEDDING(DiscriminativeFeature):
    """Embed each detected colony, or each cell of a fixed grid, with a frozen DINO ViT.

    ``MeasureDinoEmbedding`` writes one table per model and region, never
    columns of the measurement table: ``Object_Label`` (or the ``Grid_*``
    cell keys) and one ``embedding`` vector per row, under
    ``tables/<run>/embeddings/<model>/<region>/table.parquet`` in each
    image's store and ``deliverables/embeddings/<model>/<region>.parquet``
    for the run. The region is ``crops`` or ``grid``, or ``crops-<name>`` /
    ``grid-<name>`` for an embedder given a ``name``.
    """

    @classmethod
    def metric_family(cls) -> str:
        return "DinoEmbedding"

    @classmethod
    def default_unit(cls) -> UNITS:
        return UNITS.NOT_A_QUANTITY

    EMBEDDING = Entry(
        "Embedding",
        "A fixed-length float16 feature vector for one colony crop or grid cell, "
        "stored as an Arrow FixedSizeList<float16>[D] column named ``embedding``. "
        "D is the backbone's hidden size: 384 for ViT-S, 768 for ViT-B and 1024 "
        "for ViT-L. With pooling ``crop_cls`` it is the CLS token of the region's "
        "crop, resized to ``crop_px`` square; with ``crop_mean`` it is the mean of "
        "that crop's patch tokens, with the CLS and register tokens left out; with "
        "``masked_dense`` it is the mean of the plate's patch tokens under the "
        "region's mask, from one tiled pass over the whole image at native "
        "resolution. A component has no unit and no fixed range. Compare vectors "
        "from the same model, revision and pooling only, by cosine similarity or "
        "Euclidean distance, never as single values.",
    )
