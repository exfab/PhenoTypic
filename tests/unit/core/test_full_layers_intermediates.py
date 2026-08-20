"""full_layers=True writes complete OME-Zarr snapshots per node."""
import numpy as np

from phenotypic import Image
from phenotypic._core._image_pipeline import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes


def _store_layers(store) -> set[str]:
    """Every layer a store carries: its series plus its label."""
    block = read_phenotypic_attributes(store)
    return set(block.get(PhenotypicAttr.SERIES, {})) | set(
        block.get(PhenotypicAttr.LABELS, {})
    )


def test_full_layers_writes_complete_snapshots(tmp_path):
    from phenotypic.data import load_synth_yeast_plate

    image = load_synth_yeast_plate()
    pipeline = ImagePipeline(ops=[BlurGauss(sigma=1), OtsuDetector()])
    out_dir = tmp_path / "full"

    result = pipeline.apply_with_intermediates(
        image, output_dir=out_dir, full_layers=True
    )

    base = out_dir / "base_00.ome.zarr"
    assert base.is_dir()
    block = read_phenotypic_attributes(base)
    assert block[PhenotypicAttr.STORE_SCHEMA_VERSION] == 3
    # A full snapshot also carries the class, which is what lets the builder
    # reconstruct a faithful GridImage from any node.
    assert block[PhenotypicAttr.IMAGE_CLASS] == "GridImage"
    assert _store_layers(base) == {"rgb", "gray", "detect_mat", "objmap"}

    enhancer_store = out_dir / "00_BlurGauss.ome.zarr"
    assert enhancer_store.is_dir()
    # full snapshot keeps ALL layers, not just the modified detect_mat
    assert _store_layers(enhancer_store) == {"rgb", "gray", "detect_mat", "objmap"}
    assert not np.array_equal(
        Image.load_layer_zarr(enhancer_store, "detect_mat"),
        Image.load_layer_zarr(base, "detect_mat"),
    )

    detector_store = out_dir / "01_OtsuDetector.ome.zarr"
    assert detector_store.is_dir()
    assert _store_layers(detector_store) == {"rgb", "gray", "detect_mat", "objmap"}
    assert int(Image.load_layer_zarr(detector_store, "objmap").max()) > 0
    assert result.image is not None


def test_full_layers_false_keeps_delta_behavior(tmp_path):
    from phenotypic.data import load_synth_yeast_plate

    image = load_synth_yeast_plate()
    pipeline = ImagePipeline(ops=[BlurGauss(sigma=1)])
    out_dir = tmp_path / "delta"

    pipeline.apply_with_intermediates(image, output_dir=out_dir)  # default

    # The delta carries the modified layer plus the co-written `gray` primary
    # series -- and nothing else.
    assert _store_layers(out_dir / "00_BlurGauss.ome.zarr") == {"gray", "detect_mat"}
