"""Tests for FssDinoDetector (Spec 2b, Task 5).

Construction / capability / serialization / support-set validation / exemplar
defaults run WITHOUT weights (lazy load). The faithful FSSDINO algorithm
(class-specific k-means prototypes + Gram-matrix refinement + mean/max score
combination, clean-room from arXiv:2602.07550) is unit-tested on synthetic
features. ONE real DINOv2 functional test runs ``.apply()`` with the bundled
exemplar (skips only if the backbone is unavailable — NOT gated).
"""

import json

import numpy as np
import pytest

from phenotypic import ImagePipeline
from phenotypic.abc_ import GpuDetector, ObjectDetector
from phenotypic.detect.nn import FssDinoDetector


# ---------------------------------------------------------------------------
# Construction / capability / exemplar defaults
# ---------------------------------------------------------------------------


class TestFssDinoConstruction:
    def test_capabilities(self):
        det = FssDinoDetector()
        assert det.input_layer == "rgb"
        assert det.output_kind == "semantic"
        assert det.supports_batching is False

    def test_dino_version_defaults_to_2(self):
        # DINOv2 default, ungated (Spec §4.4 — prefer DINOv2 where the method
        # permits, so a gate-free functional test can run).
        assert FssDinoDetector().dino_version == 2

    def test_n_clusters_default_and_field(self):
        assert FssDinoDetector().n_clusters == 5  # paper sets n_c = 5
        assert FssDinoDetector(n_clusters=8).n_clusters == 8

    def test_feature_layer_default_is_last(self):
        # FSSDINO's "Semantic Selection Gap": last layer is the safe default.
        assert FssDinoDetector().feature_layer == -1

    def test_default_support_set_is_bundled(self):
        det = FssDinoDetector()
        assert len(det.support_images) == 1
        assert len(det.support_masks) == 1
        assert det.support_images[0].is_file()
        assert det.support_masks[0].is_file()

    def test_default_similarity_thresh(self):
        assert FssDinoDetector().similarity_thresh == 0.5

    def test_serialization_round_trip(self):
        det = FssDinoDetector(n_clusters=8, similarity_thresh=0.6, feature_layer=9)
        r = FssDinoDetector.from_json(det.to_json())
        assert r.n_clusters == 8
        assert r.similarity_thresh == 0.6
        assert r.feature_layer == 9

    def test_pipeline_json_round_trip(self):
        pipe = ImagePipeline(ops=[FssDinoDetector(n_clusters=3)])
        restored = ImagePipeline.from_json(pipe.to_json())
        det = list(restored._ops.values())[0]
        assert isinstance(det, FssDinoDetector)
        assert det.n_clusters == 3

    def test_private_attrs_not_in_json(self):
        det = FssDinoDetector()
        config = json.loads(ImagePipeline(ops=[det]).to_json())
        for op_cfg in config["pipe_cfgs"].values():
            params = op_cfg.get("params", {})
            assert "_model" not in params
            assert "_fg_prototypes" not in params

    def test_constructs_without_transformers(self):
        FssDinoDetector()  # no raise

    def test_is_gpu_and_object_detector(self):
        det = FssDinoDetector()
        assert isinstance(det, GpuDetector)
        assert isinstance(det, ObjectDetector)

    def test_missing_support_raises_on_load(self):
        det = FssDinoDetector(support_images=[], support_masks=[])
        with pytest.raises(ValueError, match="support"):
            det._ensure_model_loaded()

    def test_mismatched_support_lengths_raise_on_load(self, tmp_path):
        det = FssDinoDetector(
            support_images=[tmp_path / "a.png", tmp_path / "b.png"],
            support_masks=[tmp_path / "a_mask.png"],
        )
        with pytest.raises(ValueError, match="support"):
            det._ensure_model_loaded()


# ---------------------------------------------------------------------------
# Faithful FSSDINO algorithm (C3) — synthetic features, no model
# ---------------------------------------------------------------------------


class TestFssDinoAlgorithm:
    def test_cluster_prototypes_kmeans_cosine(self):
        from phenotypic.detect.nn._fssdino_detector import cluster_prototypes

        # Two well-separated direction clusters → 2 prototypes recover them.
        rng = np.random.default_rng(0)
        a = np.array([1.0, 0.0, 0.0]) + rng.normal(scale=0.01, size=(20, 3))
        b = np.array([0.0, 1.0, 0.0]) + rng.normal(scale=0.01, size=(20, 3))
        feats = np.vstack([a, b]).astype(np.float32)
        protos = cluster_prototypes(feats, n_clusters=2)
        assert protos.shape == (2, 3)
        # Each prototype is unit-norm and aligned with one of the two clusters.
        dirs = np.abs(protos @ np.array([[1.0, 0, 0], [0, 1.0, 0]]).T)
        assert (dirs.max(axis=1) > 0.99).all()

    def test_cluster_prototypes_caps_k_at_n_samples(self):
        from phenotypic.detect.nn._fssdino_detector import cluster_prototypes

        feats = np.eye(3, dtype=np.float32)  # 3 samples
        protos = cluster_prototypes(feats, n_clusters=10)
        assert protos.shape[0] <= 3  # cannot have more clusters than samples

    def test_gram_matrix_is_normalized_outer_mean(self):
        from phenotypic.detect.nn._fssdino_detector import gram_matrix

        # All features point along e0 → G = e0 e0^T (rank-1).
        feats = np.zeros((5, 4), np.float32)
        feats[:, 0] = 3.0  # un-normalised; gram normalises internally
        g = gram_matrix(feats)
        assert g.shape == (4, 4)
        expected = np.zeros((4, 4))
        expected[0, 0] = 1.0
        assert np.allclose(g, expected, atol=1e-5)

    def test_gram_refinement_energy_map(self):
        from phenotypic.detect.nn._fssdino_detector import gram_score_map

        # Gram from features along e0; query patches along e0 score high,
        # along e1 score ~0.
        support = np.zeros((4, 5), np.float32)
        support[:, 0] = 1.0
        g = None
        from phenotypic.detect.nn._fssdino_detector import gram_matrix

        g = gram_matrix(support)
        q = np.zeros((2, 3, 5), np.float32)
        q[0, 0, 0] = 1.0  # along e0 → high gram energy
        q[1, 1, 1] = 1.0  # along e1 → ~0
        smap = gram_score_map(q, g)
        assert smap.shape == (2, 3)
        assert smap[0, 0] > smap[1, 1]

    def test_combine_score_maps_mean_times_max(self):
        from phenotypic.detect.nn._fssdino_detector import combine_score_maps

        # Three maps; combined = mean ⊙ max (Hadamard), each map min-max
        # normalised to [0, 1] first.
        maps = [
            np.array([[0.0, 1.0], [0.5, 0.5]], np.float32),
            np.array([[0.2, 0.8], [0.4, 0.6]], np.float32),
            np.array([[0.1, 0.9], [0.3, 0.7]], np.float32),
        ]
        combined = combine_score_maps(maps)
        assert combined.shape == (2, 2)
        # The top-right cell is the max in every map → highest combined score.
        assert combined[0, 1] == combined.max()

    def test_assign_foreground_argmax_over_classes(self):
        from phenotypic.detect.nn._fssdino_detector import assign_foreground

        fg = np.array([[0.9, 0.1], [0.8, 0.2]], np.float32)
        bg = np.array([[0.1, 0.9], [0.2, 0.8]], np.float32)
        mask = assign_foreground(fg, bg, similarity_thresh=0.0)
        assert mask.dtype == bool
        assert mask[0, 0] and not mask[0, 1]  # argmax(fg, bg) per pixel
        assert mask[1, 0] and not mask[1, 1]


# ---------------------------------------------------------------------------
# Functional — real DINOv2 + bundled exemplar (NOT gated; skips if no backbone)
# ---------------------------------------------------------------------------


def _dinov2_backbone_loadable() -> bool:
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
class TestFssDinoFunctionalDinoV2:
    def test_apply_writes_objmask_on_dinov2(self, synth_plate):
        det = FssDinoDetector(
            dino_version=2, dino_size="small", n_clusters=3, device="cpu"
        )
        result = det.apply(synth_plate.copy(), inplace=False)
        objmask = result.objmask[:]
        objmap = result.objmap[:]
        assert objmask.dtype == bool
        # Semantic route: objmap auto-labels from objmask (Spec 1 §8 invariant).
        assert np.array_equal(objmap[:] > 0, objmask[:])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
