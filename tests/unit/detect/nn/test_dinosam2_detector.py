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
from pydantic import ValidationError


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
        assert det.points_per_batch == 8

    def test_points_per_batch_must_be_positive(self):
        with pytest.raises(ValidationError):
            DinoSam2Detector(points_per_batch=0)

    def test_dinov3_is_opt_in(self):
        det = DinoSam2Detector(dino_version=3, dino_size="base")
        assert det.dino_version == 3
        assert det._hf_dino_id() == "facebook/dinov3-vitb16-pretrain-lvd1689m"

    def test_dinov2_hf_id(self):
        assert DinoSam2Detector(
            dino_size="base")._hf_dino_id() == "facebook/dinov2-base"

    def test_dinov2_hf_id_small_and_large(self):
        assert DinoSam2Detector(
            dino_size="small")._hf_dino_id() == "facebook/dinov2-small"
        assert DinoSam2Detector(
            dino_size="large")._hf_dino_id() == "facebook/dinov2-large"

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
        pipe = ImagePipeline(
                ops=[DinoSam2Detector(dino_size="large", points_per_batch=4)]
        )
        restored = ImagePipeline.from_json(pipe.to_json())
        det = list(restored._ops.values())[0]
        assert isinstance(det, DinoSam2Detector)
        assert det.dino_size == "large"
        assert det.points_per_batch == 4

    def test_old_pipeline_payload_defaults_points_per_batch(self):
        config = json.loads(ImagePipeline(ops=[DinoSam2Detector()]).to_json())
        detector_config = next(
                value
                for value in config["pipe_cfgs"].values()
                if value["class"] == "DinoSam2Detector"
        )
        detector_config["params"].pop("points_per_batch")

        restored = ImagePipeline.from_json(json.dumps(config))

        assert list(restored._ops.values())[0].points_per_batch == 8

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
        from phenotypic.detect.nn._helper import _checkpoint_manager as cm
        from phenotypic.detect.nn import _sam2 as sam2_mod

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

    def test_rle_recipe_matches_boolean_recipe(self):
        from phenotypic.detect.nn._dinosam2_detector import (
            _assemble_objmap,
            _assemble_rle_objmap,
        )
        from phenotypic.detect.nn._helper._sam2_rle import encode_uncompressed_rle

        outer = np.zeros((12, 15), bool)
        outer[1:10, 1:13] = True
        inner = np.zeros((12, 15), bool)
        inner[4:7, 5:9] = True
        distinct = np.zeros((12, 15), bool)
        distinct[9:11, 12:15] = True
        masks = [inner, outer, distinct]
        scores = np.array([0.9, 0.95, 0.1])
        expected = _assemble_objmap(masks, scores, 0.5, 0.7)
        records = [
            {
                "segmentation": encode_uncompressed_rle(mask),
                "area"        : int(mask.sum()),
            }
            for mask in masks
        ]
        actual = _assemble_rle_objmap(records, scores, 0.5, 0.7, outer.shape)
        np.testing.assert_array_equal(actual, expected)

    def test_rle_merge_matches_boolean_merge(self):
        from phenotypic.detect.nn._dinosam2_detector import _merge_by_iou
        from phenotypic.detect.nn._helper._sam2_rle import (
            encode_uncompressed_rle,
            merge_rle_records_by_iou,
        )

        rng = np.random.default_rng(11)
        masks = [rng.random((10, 14)) > 0.75 for _ in range(8)]
        bool_kept = _merge_by_iou(masks, 0.2)
        records = [
            {
                "segmentation": encode_uncompressed_rle(mask),
                "area"        : int(mask.sum()),
                "identity"    : id(mask),
            }
            for mask in masks
        ]
        rle_kept = merge_rle_records_by_iou(records, 0.2)
        assert [record["identity"] for record in rle_kept] == [
            id(mask) for mask in bool_kept
        ]

    def test_rle_path_preserves_pooling_scores_order_and_final_map(self):
        from phenotypic.detect.nn._helper import _dino_support
        from phenotypic.detect.nn._dinosam2_detector import (
            _assemble_objmap,
            _assemble_rle_objmap,
            _score_by_prototype,
        )
        from phenotypic.detect.nn._helper._sam2_rle import (
            decode_uncompressed_rle,
            encode_uncompressed_rle,
        )
        from phenotypic.detect.nn._helper._tiling import _Tile

        shape = (12, 15)
        outer = np.zeros(shape, bool)
        outer[1:10, 1:13] = True
        inner = np.zeros(shape, bool)
        inner[4:7, 5:9] = True
        distinct = np.zeros(shape, bool)
        distinct[9:11, 12:15] = True
        masks = [inner, outer, distinct]
        predicted_ious = [0.9, 0.95, 0.2]
        yy, xx = np.indices(shape)
        dense = np.stack((yy + 1, xx + 1, yy + xx + 1), axis=-1).astype(
                np.float32
        )
        tiles = [_Tile(0, 0, *shape)]

        bool_features = np.stack(
                [
                    _dino_support.pool_prototype_tiled([dense], tiles, mask, 1)
                    for mask in masks
                ]
        )
        records = [
            {
                "segmentation" : encode_uncompressed_rle(mask),
                "area"         : int(mask.sum()),
                "predicted_iou": predicted_iou,
            }
            for mask, predicted_iou in zip(masks, predicted_ious)
        ]
        rle_features = np.stack(
                [
                    _dino_support.pool_prototype_tiled(
                            [dense],
                            tiles,
                            decode_uncompressed_rle(record["segmentation"]),
                            1,
                    )
                    for record in records
                ]
        )
        np.testing.assert_array_equal(rle_features, bool_features)

        detector = DinoSam2Detector()
        bool_prototype = detector._foreground_prototype(bool_features, records)
        rle_prototype = detector._foreground_prototype(rle_features, records)
        bool_scores = _score_by_prototype(bool_features, bool_prototype)
        rle_scores = _score_by_prototype(rle_features, rle_prototype)
        np.testing.assert_array_equal(rle_scores, bool_scores)

        expected = _assemble_objmap(masks, bool_scores, 0.0, 0.7)
        actual = _assemble_rle_objmap(records, rle_scores, 0.0, 0.7, shape)
        np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# F3 — tiled DINO features + crop-pyramid pass-through (no weights)
# ---------------------------------------------------------------------------


class TestDinoSam2Tiling:
    def test_has_tiling_fields(self):
        det = DinoSam2Detector()
        assert det.tile_px == 518  # 14 * 37, an exact DINOv2 patch multiple
        assert det.tile_overlap == 0.15
        assert det.crop_n_layers == 1
        assert det.crop_nms_thresh == 0.7
        assert det.crop_overlap_ratio == 512 / 1500
        assert det.crop_n_points_downscale_factor == 1

    def test_pool_prototype_tiled_is_nonzero_for_a_small_colony(self):
        """F3 regression: on a full plate a 30px colony is 0.16 patches wide,
        so pool_prototype rounds it to empty and returns a zero vector."""
        from phenotypic.detect.nn._helper._dino_support import pool_prototype_tiled
        from phenotypic.detect.nn._helper._tiling import _Tile

        tiles = [_Tile(0, 0, 518, 518)]
        dense = [np.ones((37, 37, 8), dtype=np.float32)]
        mask = np.zeros((518, 518), dtype=bool)
        mask[250:280, 250:280] = True  # 30 px colony

        proto = pool_prototype_tiled(dense, tiles, mask, 14)
        assert proto.shape == (8,)
        assert np.any(proto)  # NOT the zero vector

    def test_pool_prototype_tiled_pools_from_the_owning_tile(self):
        """The prototype must come from the tile whose core holds the centroid."""
        from phenotypic.detect.nn._helper._dino_support import pool_prototype_tiled
        from phenotypic.detect.nn._helper._tiling import _plan_tiles

        tiles = _plan_tiles((518, 800), 518, 0.15)
        assert len(tiles) == 2
        dense = [
            np.ones((37, 37, 4), dtype=np.float32),
            np.full((37, 37, 4), 2.0, dtype=np.float32),
        ]

        left = np.zeros((518, 800), dtype=bool)
        left[250:280, 100:130] = True
        right = np.zeros((518, 800), dtype=bool)
        right[250:280, 600:630] = True

        assert np.allclose(pool_prototype_tiled(dense, tiles, left, 14), 1.0)
        assert np.allclose(pool_prototype_tiled(dense, tiles, right, 14), 2.0)

    def test_whole_plate_pooling_collapses_where_tiled_pooling_does_not(self):
        """The F3 mechanism, at plate scale and without a backbone.

        A plate reaching the ViT at the 224-px classification preset gives a
        16x16 grid whatever the plate's size, so a 30 px colony is a fraction
        of one patch, rounds to empty, and pool_prototype returns its
        zero-vector fail-safe. The same colony inside a 518 px tile spans
        30 / 14 = 2.1 patches and pools normally.
        """
        from phenotypic.detect.nn._helper._dino_support import (
            pool_prototype,
            pool_prototype_tiled,
        )
        from phenotypic.detect.nn._helper._tiling import _plan_tiles

        shape = (1500, 2000)
        mask = np.zeros(shape, dtype=bool)
        mask[1000:1030, 1200:1230] = True  # a 30 px colony

        whole_grid = np.ones((16, 16, 4), dtype=np.float32)  # the 224 preset
        # proc_hw is the geometry the plate actually reached the ViT at.
        assert not np.any(
                pool_prototype(whole_grid, mask, proc_hw=(224, 224), patch=14)
        )

        tiles = _plan_tiles(shape, 518, 0.15)
        dense = [np.ones((37, 37, 4), dtype=np.float32) for _ in tiles]
        assert np.any(pool_prototype_tiled(dense, tiles, mask, 14))

    def test_pool_prototype_tiled_empty_mask_is_the_zero_fail_safe(self):
        from phenotypic.detect.nn._helper._dino_support import pool_prototype_tiled
        from phenotypic.detect.nn._helper._tiling import _Tile

        tiles = [_Tile(0, 0, 518, 518)]
        dense = [np.ones((37, 37, 8), dtype=np.float32)]
        proto = pool_prototype_tiled(
                dense, tiles, np.zeros((518, 518), dtype=bool), 14
        )
        assert proto.shape == (8,)
        assert not np.any(proto)

    def test_crop_fields_reach_build_sam2_generator(self, monkeypatch):
        from phenotypic.detect.nn._helper import _checkpoint_manager as cm
        from phenotypic.detect.nn import _sam2 as sam2_mod

        seen: dict = {}
        monkeypatch.setattr(cm, "resolve_device", lambda device: "cpu")
        monkeypatch.setattr(
                sam2_mod,
                "build_sam2_generator",
                lambda *a, **k: (seen.update(k), object())[1],
        )

        class _FakeModel:
            def to(self, device):
                return self

        monkeypatch.setitem(
                sys.modules,
                "transformers",
                types.SimpleNamespace(
                        AutoModel=types.SimpleNamespace(
                                from_pretrained=lambda repo_id: _FakeModel()
                        ),
                        AutoImageProcessor=types.SimpleNamespace(
                                from_pretrained=lambda repo_id: object()
                        ),
                ),
        )

        det = DinoSam2Detector(
                device="cpu",
                crop_n_layers=2,
                crop_nms_thresh=0.55,
                crop_overlap_ratio=0.25,
                crop_n_points_downscale_factor=2,
        )
        det._ensure_model_loaded()

        assert seen["crop_n_layers"] == 2
        assert seen["crop_nms_thresh"] == 0.55
        assert seen["crop_overlap_ratio"] == 0.25
        assert seen["crop_n_points_downscale_factor"] == 2
        assert seen["min_mask_region_area"] == 100
        assert seen["points_per_batch"] == 8

    def test_infer_one_extracts_features_per_tile_not_per_plate(self, monkeypatch):
        """_infer_one must never hand the whole plate to the ViT: on a
        600x800 plate at tile_px=518 that is four 518x518 crops."""
        from phenotypic.detect.nn._helper import _dino_support

        det = DinoSam2Detector(device="cpu")
        proposal = np.zeros((600, 800), dtype=bool)
        proposal[300:340, 400:440] = True

        det._device = "cpu"
        det._dino_processor = object()
        det._dino_model = types.SimpleNamespace(
                config=types.SimpleNamespace(patch_size=14, num_register_tokens=0)
        )
        det._generator = types.SimpleNamespace(
                generate=lambda rgb: [
                    {"segmentation": proposal, "predicted_iou": 0.9},
                    {"segmentation": ~proposal, "predicted_iou": 0.8},
                ]
        )

        shapes: list = []

        def fake_extract(model, processor, rgb, *, device):
            shapes.append(rgb.shape[:2])
            return np.ones((rgb.shape[0] // 14, rgb.shape[1] // 14, 6), np.float32)

        monkeypatch.setattr(_dino_support, "extract_patch_features", fake_extract)

        objmap = det._infer_one(np.zeros((600, 800, 3), dtype=np.uint8))

        assert objmap.shape == (600, 800)
        assert shapes == [(518, 518)] * 4
        assert (600, 800) not in shapes


# ---------------------------------------------------------------------------
# Functional — requires foundation + sam2 + weights (skips otherwise)
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
class TestDinoSam2FunctionalDinoV2:
    def test_prototypes_are_not_all_zero(self, synth_plate):
        """Direct F3 regression: before the fix every proposal pooled an empty
        mask and got the zero vector, so all scores were identical.

        ``synth_plate`` is only 600x800, so it cannot reproduce the collapse on
        its own -- the whole-image grid is already 42x57. The plate-scale
        mechanism is covered by
        ``TestDinoSam2Tiling.test_whole_plate_pooling_collapses_where_tiled_pooling_does_not``.
        What this asserts is that the tiled path pools real DINOv2 features
        that are non-zero *and* colony-specific.
        """
        from transformers import AutoImageProcessor, AutoModel

        from phenotypic.detect.nn._helper._dino_support import (
            extract_patch_features,
            pool_prototype_tiled,
        )
        from phenotypic.detect.nn._helper._tiling import _plan_tiles

        model = AutoModel.from_pretrained("facebook/dinov2-small").eval()
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        patch = int(model.config.patch_size)

        rgb = np.asarray(synth_plate.rgb[:], dtype=np.uint8)
        objmap = np.asarray(synth_plate.objmap[:])
        labels = np.unique(objmap)[1:6]

        tiles = _plan_tiles(rgb.shape[:2], 518, 0.15)
        dense = [
            extract_patch_features(
                    model, processor, rgb[t.y0:t.y1, t.x0:t.x1], device="cpu"
            )
            for t in tiles
        ]
        protos = [
            pool_prototype_tiled(dense, tiles, objmap == lab, patch)
            for lab in labels
        ]

        print(f"tiles={len(tiles)} dense_grid={dense[0].shape} patch={patch}")
        print(f"norms={[round(float(np.linalg.norm(p)), 3) for p in protos]}")
        print(f"distinct={len({tuple(np.round(p, 4)) for p in protos})}/{len(protos)}")

        assert all(np.any(pr) for pr in protos)
        assert len({tuple(np.round(pr, 4)) for pr in protos}) > 1


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
