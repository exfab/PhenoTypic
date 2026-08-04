"""full_layers=True writes complete v2 HDF snapshots per node."""
import h5py
from phenotypic._core._image_pipeline import ImagePipeline
from phenotypic.enhance import BlurGauss
from phenotypic.detect import OtsuDetector


def test_full_layers_writes_v2_snapshots(tmp_path):
    from phenotypic.data import load_synth_yeast_plate

    image = load_synth_yeast_plate()
    pipeline = ImagePipeline(ops=[BlurGauss(sigma=1), OtsuDetector()])
    out_dir = tmp_path / "full"

    result = pipeline.apply_with_intermediates(
        image, output_dir=out_dir, full_layers=True
    )

    base = out_dir / "base_00.h5"
    assert base.exists()
    with h5py.File(base, "r") as f:
        assert int(f.attrs["schema_version"]) == 2
        assert "layers" in f
        for layer in ("gray", "detect_mat", "objmap"):
            assert layer in f["layers"]

    enhancer_file = out_dir / "00_BlurGauss.h5"
    assert enhancer_file.exists()
    with h5py.File(enhancer_file, "r") as f:
        assert "layers" in f
        # full snapshot keeps ALL layers, not just the modified detect_mat
        assert "gray" in f["layers"]
        assert "detect_mat" in f["layers"]
        assert "objmap" in f["layers"]

    detector_file = out_dir / "01_OtsuDetector.h5"
    assert detector_file.exists()
    with h5py.File(detector_file, "r") as f:
        assert "objmap" in f["layers"]
    assert result.image is not None


def test_full_layers_false_keeps_delta_behavior(tmp_path):
    from phenotypic.data import load_synth_yeast_plate

    image = load_synth_yeast_plate()
    pipeline = ImagePipeline(ops=[BlurGauss(sigma=1)])
    out_dir = tmp_path / "delta"

    pipeline.apply_with_intermediates(image, output_dir=out_dir)  # default

    with h5py.File(out_dir / "00_BlurGauss.h5", "r") as f:
        # legacy flat layout, only the modified layer
        assert "detect_mat" in f
        assert "layers" not in f
        assert "rgb" not in f
