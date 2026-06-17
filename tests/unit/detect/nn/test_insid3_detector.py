"""Tests for Insid3Detector (Spec 2b, Task 4).

Construction / capability / serialization / exemplar-defaults / semantic-route
run WITHOUT weights (lazy load). The faithful INSID3 positional-bias-removal
math (SVD positional basis → orthogonal-complement projection) is unit-tested
on synthetic features. ONE real DINOv2 functional test loads the bundled
exemplar and runs ``.apply()`` (skips only if transformers / the backbone are
unavailable — NOT gated).
"""

import json

import numpy as np
import pytest

from phenotypic import ImagePipeline
from phenotypic.abc_ import GpuDetector, ObjectDetector
from phenotypic.detect.nn import Insid3Detector


# ---------------------------------------------------------------------------
# Construction / capability / exemplar defaults
# ---------------------------------------------------------------------------


class TestInsid3Construction:
    def test_capabilities(self):
        det = Insid3Detector()
        assert det.input_layer == "rgb"
        assert det.output_kind == "semantic"
        assert det.supports_batching is False

    def test_dino_version_defaults_to_3(self):
        # INSID3 is DINOv3-native (the debias targets DINOv3's positional bias).
        assert Insid3Detector().dino_version == 3

    def test_default_exemplar_is_bundled(self):
        det = Insid3Detector()
        assert det.reference_image is not None
        assert det.reference_mask is not None
        assert det.reference_image.is_file()
        assert det.reference_mask.is_file()

    def test_default_similarity_thresh(self):
        assert Insid3Detector().similarity_thresh == 0.5

    def test_svd_components_default(self):
        assert Insid3Detector().svd_components == 4

    def test_serialization_round_trip(self, tmp_path):
        det = Insid3Detector(
            reference_image=tmp_path / "r.tiff",
            reference_mask=tmp_path / "m.png",
            similarity_thresh=0.7,
            svd_components=6,
        )
        r = Insid3Detector.from_json(det.to_json())
        assert r.similarity_thresh == 0.7
        assert r.svd_components == 6
        assert str(r.reference_image).endswith("r.tiff")

    def test_pipeline_json_round_trip(self):
        pipe = ImagePipeline(ops=[Insid3Detector(similarity_thresh=0.6)])
        restored = ImagePipeline.from_json(pipe.to_json())
        det = list(restored._ops.values())[0]
        assert isinstance(det, Insid3Detector)
        assert det.similarity_thresh == 0.6

    def test_private_attrs_not_in_json(self):
        det = Insid3Detector()
        config = json.loads(ImagePipeline(ops=[det]).to_json())
        for op_cfg in config["pipe_cfgs"].values():
            params = op_cfg.get("params", {})
            assert "_model" not in params
            assert "_prototype" not in params

    def test_constructs_without_transformers(self):
        Insid3Detector()  # no raise

    def test_is_gpu_and_object_detector(self):
        det = Insid3Detector()
        assert isinstance(det, GpuDetector)
        assert isinstance(det, ObjectDetector)

    def test_missing_reference_raises_on_load(self):
        det = Insid3Detector(reference_image=None, reference_mask=None)
        with pytest.raises(ValueError, match="reference"):
            det._ensure_model_loaded()


# ---------------------------------------------------------------------------
# Faithful INSID3 positional-bias removal (C2) — synthetic features, no model
# ---------------------------------------------------------------------------


class TestPositionalDebias:
    def test_positional_basis_is_top_svd_left_singular_vectors(self):
        from phenotypic.detect.nn._insid3_detector import positional_basis

        # Build channel-direction-correlated features so SVD has a clear top
        # component: every patch is a multiple of a fixed direction u0.
        D = 8
        rng = np.random.default_rng(0)
        u0 = np.zeros(D, np.float32)
        u0[0] = 1.0
        feats = np.outer(rng.normal(size=(5 * 6)), u0).reshape(5, 6, D)
        basis = positional_basis(feats, n_components=1)
        assert basis.shape == (D, 1)
        # The recovered basis direction is (anti)parallel to u0.
        assert abs(float(np.abs(basis[:, 0] @ u0))) == pytest.approx(1.0, abs=1e-5)

    def test_debias_projects_onto_orthogonal_complement(self):
        from phenotypic.detect.nn._insid3_detector import (
            debias_features,
            positional_basis,
        )

        D = 6
        rng = np.random.default_rng(3)
        # e0 = a clear high-variance positional direction (random per-patch
        # magnitude); e1 = a small fixed semantic foreground signal.
        feats = np.zeros((4, 4, D), np.float32)
        feats[..., 0] = rng.normal(scale=5.0, size=(4, 4))  # dominant variance
        feats[1:3, 1:3, 1] = 0.5  # semantic foreground in e1
        basis = positional_basis(feats, n_components=1)
        # The recovered positional direction is ~e0 (the dominant variance axis).
        assert abs(float(basis[0, 0])) == pytest.approx(1.0, abs=1e-2)
        deb = debias_features(feats, basis)
        # The exact mathematical contract: every debiased patch is orthogonal
        # to the removed positional basis (projection onto its complement).
        flat = deb.reshape(-1, D)
        proj = flat @ basis  # (N, 1) — should be ~0 (float32 SVD precision)
        assert np.allclose(proj, 0.0, atol=1e-4)
        # Foreground patches retain their e1 (semantic) content (it dominates
        # after the positional e0 axis is removed); nothing is NaN.
        assert np.abs(deb[1, 1, 1]) > 0.5
        assert np.isfinite(deb).all()

    def test_debias_zero_components_preserves_direction(self):
        from phenotypic.detect.nn._insid3_detector import (
            debias_features,
            positional_basis,
        )

        # svd_components=0 → empty basis → debias is identity-up-to-L2-norm
        # (INSID3 always L2-normalises the debiased features).
        feats = np.random.default_rng(1).normal(size=(3, 3, 5)).astype(np.float32)
        basis = positional_basis(feats, n_components=0)
        assert basis.shape == (5, 0)
        deb = debias_features(feats, basis)
        flat_in = feats.reshape(9, 5)
        flat_out = deb.reshape(9, 5)
        # Same direction (cosine 1), unit norm.
        for i in range(9):
            cos = (flat_in[i] @ flat_out[i]) / (
                np.linalg.norm(flat_in[i]) * np.linalg.norm(flat_out[i])
            )
            assert cos == pytest.approx(1.0, abs=1e-5)
            assert np.linalg.norm(flat_out[i]) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Functional — real DINOv2 + bundled exemplar (NOT gated; skips if no backbone)
# ---------------------------------------------------------------------------


def _dinov2_backbone_loadable() -> bool:
    """True if transformers is present AND a DINOv2 backbone can be obtained.

    DINOv2 is ungated; this loads ``facebook/dinov2-small`` (downloads on first
    run if the node has network, else loads from the HF cache). Skips cleanly
    when neither is possible so the suite stays green offline.
    """
    from phenotypic.detect.nn import FOUNDATION_AVAILABLE

    if not FOUNDATION_AVAILABLE:
        return False
    try:
        from transformers import AutoImageProcessor, AutoModel

        AutoModel.from_pretrained("facebook/dinov2-small")
        AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _dinov2_backbone_loadable(),
    reason="Requires transformers + a loadable DINOv2 backbone (ungated)",
)
class TestInsid3FunctionalDinoV2:
    def test_apply_writes_objmask_on_dinov2(self, synth_plate):
        # DINOv2 path (ungated) with the bundled exemplar; the debias is a
        # near-no-op on DINOv2 (0 register tokens) but the plumbing must run.
        # similarity_thresh=0.0 is a permissive plumbing floor (the default 0.5
        # cosine floor is conservative for DINOv2-small); this asserts the full
        # apply() pipeline produces a non-empty semantic mask, not accuracy.
        det = Insid3Detector(
            dino_version=2, dino_size="small", similarity_thresh=0.0,
            device="cpu",
        )
        result = det.apply(synth_plate.copy(), inplace=False)
        objmask = result.objmask[:]
        objmap = result.objmap[:]
        assert objmask.dtype == bool
        # The detector must actually segment something on the bundled exemplar.
        assert objmask.any(), "Insid3Detector produced an empty objmask"
        # Semantic route: objmap auto-labels from objmask (Spec 1 §8 invariant).
        assert np.array_equal(objmap[:] > 0, objmask[:])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
