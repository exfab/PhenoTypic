"""Tests for DinoSam2Detector (Spec 2a, Task 6).

Construction / serialization / capability / dino-version routing run WITHOUT
weights (lazy load). The recipe-algorithm test (C2) drives the clean-room
scoring+merge on synthetic features. Functional ``.apply()`` requires
foundation + sam2 + weights and skips otherwise.
"""

import json
import sys
import types

import numpy as np
import pytest

from phenotypic import ImagePipeline
from phenotypic.abc_ import GpuDetector, ObjectDetector
from phenotypic.detect.nn import DinoSam2Detector


# ---------------------------------------------------------------------------
# Construction / capability / dino routing
# ---------------------------------------------------------------------------


class TestDinoSam2Construction:
    def test_capability_fields(self):
        det = DinoSam2Detector()
        assert det.input_layer == "rgb"
        assert det.output_kind == "instance"
        assert det.supports_batching is False

    def test_dino_version_defaults_to_2(self):
        assert DinoSam2Detector().dino_version == 2  # DINOv2, ungated (O1)

    def test_dino_size_default(self):
        assert DinoSam2Detector().dino_size == "base"

    def test_sam2_size_default_is_tiny(self):
        assert DinoSam2Detector().sam2_model_size == "tiny"

    def test_default_thresholds(self):
        det = DinoSam2Detector()
        assert det.similarity_thresh == 0.5
        assert det.merge_iou_thresh == 0.7
        assert det.min_proposal_area == 100

    def test_dinov3_is_opt_in(self):
        det = DinoSam2Detector(dino_version=3, dino_size="base")
        assert det.dino_version == 3
        assert det._hf_dino_id() == "facebook/dinov3-vitb16-pretrain-lvd1689m"

    def test_dinov2_hf_id(self):
        assert DinoSam2Detector(dino_size="base")._hf_dino_id() == "facebook/dinov2-base"

    def test_dinov2_hf_id_small_and_large(self):
        assert DinoSam2Detector(dino_size="small")._hf_dino_id() == "facebook/dinov2-small"
        assert DinoSam2Detector(dino_size="large")._hf_dino_id() == "facebook/dinov2-large"

    def test_serialization_round_trip(self):
        d = DinoSam2Detector(dino_version=3, similarity_thresh=0.6)
        r = DinoSam2Detector.from_json(d.to_json())
        assert r.dino_version == 3 and r.similarity_thresh == 0.6

    def test_constructs_without_transformers(self):
        DinoSam2Detector()  # no raise

    def test_is_gpu_and_object_detector(self):
        det = DinoSam2Detector()
        assert isinstance(det, GpuDetector)
        assert isinstance(det, ObjectDetector)


class TestDinoSam2Serialization:
    def test_pipeline_json_round_trip(self):
        pipe = ImagePipeline(ops=[DinoSam2Detector(dino_size="large")])
        restored = ImagePipeline.from_json(pipe.to_json())
        det = list(restored._ops.values())[0]
        assert isinstance(det, DinoSam2Detector)
        assert det.dino_size == "large"

    def test_private_attrs_not_in_json(self):
        det = DinoSam2Detector()
        config = json.loads(ImagePipeline(ops=[det]).to_json())
        for op_cfg in config["pipe_cfgs"].values():
            params = op_cfg.get("params", {})
            assert "_generator" not in params
            assert "_dino_model" not in params


# ---------------------------------------------------------------------------
# dino_version=3 load path (W2 — Spec 2b completes the v3 route; the 2a
# NotImplementedError stub is removed). Mock the gated snapshot pull + the
# SAM2 generator + transformers AutoModel so no weights / GPU are needed.
# ---------------------------------------------------------------------------


class TestDinoV3LoadPath:
    def test_dinov3_load_routes_through_gated_manager(self, monkeypatch):
        from phenotypic.detect.nn import _checkpoint_manager as cm
        from phenotypic.detect.nn import _sam2_detector as sam2_mod

        # Accept the gate; stub the gated snapshot pull (no network).
        monkeypatch.setenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", "dinov3")
        pulled: dict = {}
        monkeypatch.setattr(
            cm, "snapshot_download",
            lambda **kw: pulled.update(kw) or "/fake/cache/dinov3",
        )
        monkeypatch.setattr(cm, "resolve_device", lambda device: "cpu")

        # Stub the SAM2 generator + transformers load at their real homes so
        # nothing real loads (the detector imports both inside the method).
        monkeypatch.setattr(
            sam2_mod, "build_sam2_generator", lambda *a, **k: object()
        )

        class _FakeModel:
            def to(self, device):
                return self

        seen: dict = {}

        class _FakeAutoModel:
            @classmethod
            def from_pretrained(cls, repo_id):
                seen["model"] = repo_id
                return _FakeModel()

        class _FakeAutoImageProcessor:
            @classmethod
            def from_pretrained(cls, repo_id):
                return object()

        # Stub the module import itself. On Windows the real Transformers
        # AutoModel placeholder requires torch before it can be monkeypatched.
        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(
                AutoModel=_FakeAutoModel,
                AutoImageProcessor=_FakeAutoImageProcessor,
            ),
        )

        det = DinoSam2Detector(dino_version=3, dino_size="base", device="cpu")
        det._ensure_model_loaded()  # no NotImplementedError, no network

        # The gated DINOv3 snapshot was pulled for the right repo id, and the
        # backbone was loaded from that id.
        assert pulled["repo_id"] == "facebook/dinov3-vitb16-pretrain-lvd1689m"
        assert seen["model"] == "facebook/dinov3-vitb16-pretrain-lvd1689m"
        assert det._dino_model is not None


# ---------------------------------------------------------------------------
# Recipe algorithm (C2) — synthetic features, no model
# ---------------------------------------------------------------------------


class TestRecipeAlgorithm:
    def test_score_proposals_by_prototype_cosine(self):
        from phenotypic.detect.nn._dinosam2_detector import _score_by_prototype

        # Three proposal features: two foreground-like (aligned), one
        # background-like (anti-aligned). Prototype = mean of high-confidence
        # (here, all) — but the outlier should score low.
        feats = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [-1.0, 0.0, 0.0],  # background, opposite direction
            ],
            dtype=np.float64,
        )
        # Prototype from the two foreground proposals.
        prototype = feats[:2].mean(axis=0)
        scores = _score_by_prototype(feats, prototype)
        assert scores[0] > 0.8 and scores[1] > 0.8
        assert scores[2] < 0.0  # anti-aligned → negative cosine

    def test_iou_merge_dedups_near_duplicates(self):
        from phenotypic.detect.nn._dinosam2_detector import _merge_by_iou

        a = np.zeros((20, 20), bool)
        a[2:8, 2:8] = True
        b = np.zeros((20, 20), bool)
        b[2:8, 2:8] = True  # duplicate
        c = np.zeros((20, 20), bool)
        c[12:18, 12:18] = True  # distinct
        kept = _merge_by_iou([a, b, c], iou_thresh=0.7)
        assert len(kept) == 2  # one of {a, b} dropped, c survives

    def test_recipe_assembles_objmap_from_scored_proposals(self):
        from phenotypic.detect.nn._dinosam2_detector import _assemble_objmap

        fg = np.zeros((20, 20), bool)
        fg[2:8, 2:8] = True
        bg = np.zeros((20, 20), bool)
        bg[12:18, 12:18] = True
        # fg scores above threshold, bg below.
        objmap = _assemble_objmap(
            proposals=[fg, bg],
            scores=np.array([0.9, 0.1]),
            similarity_thresh=0.5,
            merge_iou_thresh=0.7,
        )
        assert objmap.dtype == np.uint16
        assert objmap.max() == 1  # only the foreground proposal survives
        assert objmap[4, 4] == 1
        assert objmap[14, 14] == 0  # background dropped


# ---------------------------------------------------------------------------
# Functional — requires foundation + sam2 + weights (skips otherwise)
# ---------------------------------------------------------------------------


def _dinosam2_runnable() -> bool:
    import importlib.util

    from phenotypic.detect.nn import FOUNDATION_AVAILABLE

    if not FOUNDATION_AVAILABLE:
        return False
    return importlib.util.find_spec("sam2") is not None


@pytest.mark.skipif(
    not _dinosam2_runnable(),
    reason="Requires phenotypic[foundation] + sam2 + cached weights",
)
class TestDinoSam2Functional:
    def test_constructs_generator_fields(self):
        # Even when sam2 is present, weight download may be unavailable; this
        # only checks the detector builds without raising at construction.
        det = DinoSam2Detector(device="cpu")
        assert det._generator is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
