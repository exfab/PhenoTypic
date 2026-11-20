"""``ImagePipeline.measure()`` skips embedders; ``embed()`` runs them (#281 P4 T1, spec E2)."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize
from phenotypic.schema import OBJECT
from phenotypic.sdk_._embedding_tables import EmbeddingPayload


def test_measure_skips_and_embed_runs(toy_embedder, plate):
    pipe = ImagePipeline(ops=[OtsuDetector()], meas={"size": MeasureSize(), "emb": toy_embedder})
    img = pipe.apply(plate)
    assert not any("embedding" in c for c in pipe.measure(img).columns)
    payloads = pipe.embed(img)
    assert list(payloads) == ["toy-v1/rgb/objects"]
    assert len(payloads["toy-v1/rgb/objects"].frame) == img.num_objects


def test_two_embedders_with_one_payload_key_are_refused(toy_embedder):
    twin = toy_embedder.model_copy()                  # same model and region, e.g. another pooling
    with pytest.raises(ValueError, match="emb_a.*emb_b.*toy-v1/rgb/objects"):
        ImagePipeline(ops=[OtsuDetector()], meas={"emb_a": toy_embedder, "emb_b": twin})


def test_a_crop_and_a_grid_embedder_of_one_model_coexist(toy_embedder, toy_grid_embedder, plate):
    pipe = ImagePipeline(ops=[OtsuDetector()], meas={"crops": toy_embedder, "grid": toy_grid_embedder})
    payloads = pipe.embed(pipe.apply(plate))
    assert list(payloads) == ["toy-v1/rgb/objects", "toy-v1/rgb/grid"]


def test_measure_without_any_plain_measurer_returns_the_info_block(toy_embedder, plate):
    pipe = ImagePipeline(ops=[OtsuDetector()], meas={"emb": toy_embedder})
    frame = pipe.measure(pipe.apply(plate))
    assert str(OBJECT.LABEL) in frame.columns
    assert "embedding" not in frame.columns


def test_a_pipeline_without_embedders_embeds_nothing(plate):
    pipe = ImagePipeline(ops=[OtsuDetector()], meas={"size": MeasureSize()})
    assert pipe.embed(pipe.apply(plate)) == {}


def test_assigning_a_duplicate_embedder_later_is_refused(toy_embedder):
    pipe = ImagePipeline(ops=[OtsuDetector()], meas={"emb_a": toy_embedder})
    with pytest.raises(ValueError, match="emb_a.*emb_b.*toy-v1/rgb/objects"):
        pipe.set_meas({"emb_a": toy_embedder, "emb_b": toy_embedder.model_copy()})


def test_the_crop_payload_descriptor_binds_the_objmap(toy_embedder, detected_plate):
    from phenotypic.sdk_._measurement_tables import objmap_digest

    payload = ImagePipeline(meas={"emb": toy_embedder}).embed(detected_plate)["toy-v1/rgb/objects"]
    assert isinstance(payload, EmbeddingPayload)
    descriptor = payload.descriptor
    assert descriptor["key"] == str(OBJECT.LABEL)
    assert descriptor["dim"] == 4
    assert descriptor["input"] == {"series": "rgb", "level": 0, "layer": "rgb"}
    assert descriptor["producer"]["class"] == type(toy_embedder).__name__
    assert len(descriptor["producer"]["params_sha256"]) == 64
    assert descriptor["binds"] == {
        "objmap_path": "rgb/labels/objmap",
        "objmap_sha256": objmap_digest(detected_plate.objmap[:]),
    }


def test_the_grid_payload_descriptor_binds_the_level0_array(toy_grid_embedder, plate):
    import numpy as np

    from phenotypic.sdk_._measurement_tables import array_digest

    payload = ImagePipeline(meas={"g": toy_grid_embedder}).embed(plate)["toy-v1/rgb/grid"]
    binds = payload.descriptor["binds"]
    assert set(binds) == {"array_sha256", "grid_sha256"}
    assert binds["array_sha256"] == array_digest(np.moveaxis(plate.rgb[:], -1, 0))
    assert payload.descriptor["key"] == "Grid_RowMajorIdx"


def test_two_named_grid_embedders_of_one_model_coexist(plate):
    """#281 named grid regions: rig1's ``wells`` and ``agar`` grids, one model."""
    from tests._tiny_run import named_grid_embedders, toy_grid_embedder_of

    pipe = ImagePipeline(meas=named_grid_embedders(toy_grid_embedder_of))
    payloads = pipe.embed(plate)
    assert list(payloads) == ["toy-v1/rgb/grid-wells", "toy-v1/rgb/grid-agar"]
    assert [len(p.frame) for p in payloads.values()] == [4, 1]


def test_a_named_and_an_unnamed_grid_of_one_model_coexist():
    from tests._tiny_run import toy_grid_embedder_of

    ImagePipeline(meas={"a": toy_grid_embedder_of(), "b": toy_grid_embedder_of(name="wells")})


def test_two_named_crop_embedders_of_one_model_coexist(detected_plate):
    """#281 named regions: two poolings of the colonies, one model, two tables."""
    from phenotypic.measure import MeasureDinoEmbedding
    from tests._tiny_run import toy_embedder_class

    pipe = ImagePipeline(
        meas={
            "cls": MeasureDinoEmbedding(pooling="crop_cls", name="cls"),
            "mean": MeasureDinoEmbedding(pooling="crop_mean", name="mean"),
        }
    )
    assert [m.payload_key() for m in pipe._meas.values()] == [
        "dinov2-vitb14/rgb/objects-cls", "dinov2-vitb14/rgb/objects-mean",
    ]
    with pytest.raises(ValueError, match="dinov2-vitb14/rgb/objects"):
        ImagePipeline(
            meas={
                "cls": MeasureDinoEmbedding(pooling="crop_cls"),
                "mean": MeasureDinoEmbedding(pooling="crop_mean"),
            }
        )
    toy = toy_embedder_class()
    payloads = ImagePipeline(meas={"a": toy(name="a"), "b": toy(name="b")}).embed(detected_plate)
    assert list(payloads) == ["toy-v1/rgb/objects-a", "toy-v1/rgb/objects-b"]
    assert all(len(p.frame) == detected_plate.num_objects for p in payloads.values())


def test_two_embedders_with_one_name_are_refused():
    from tests._tiny_run import toy_grid_embedder_of

    with pytest.raises(ValueError, match="wells_a.*wells_b.*toy-v1/rgb/grid-wells"):
        ImagePipeline(
            meas={
                "wells_a": toy_grid_embedder_of(name="wells"),
                "wells_b": toy_grid_embedder_of(name="wells"),
            }
        )


def test_an_rgb_and_a_gray_copy_with_one_name_coexist(plate):
    """#281 layer folder: the layer is its own key segment, so ``wells`` on rgb
    and ``wells`` on gray are two tables without a second name."""
    from tests._tiny_run import toy_grid_embedder_of

    pipe = ImagePipeline(
        meas={
            "wells_rgb": toy_grid_embedder_of(name="wells"),
            "wells_gray": toy_grid_embedder_of(name="wells", layer="gray"),
        }
    )
    payloads = pipe.embed(plate)
    assert list(payloads) == ["toy-v1/rgb/grid-wells", "toy-v1/gray/grid-wells"]
    rgb, gray = (np.stack(p.frame["embedding"]) for p in payloads.values())
    assert not np.array_equal(rgb, gray)


def test_two_gray_copies_with_one_name_are_refused():
    from tests._tiny_run import toy_grid_embedder_of

    with pytest.raises(ValueError, match="a.*b.*toy-v1/gray/grid-wells"):
        ImagePipeline(
            meas={
                "a": toy_grid_embedder_of(name="wells", layer="gray"),
                "b": toy_grid_embedder_of(name="wells", layer="gray"),
            }
        )
