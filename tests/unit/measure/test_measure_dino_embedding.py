"""``MeasureDinoEmbedding``: DINO crop and grid embeddings (#281 P4 Task 3, spec E3).

Every test but the ``slow`` smoke test runs on ``stub_backbone``, a tiny
deterministic stand-in for DINOv2 (``tests/_tiny_run.py``): no download, no
GPU. Tests that run the stub need torch and skip without it; tests refused
before the backbone loads do not.
"""

from __future__ import annotations

import importlib.util
import json
import time

import numpy as np
import pytest

from phenotypic import Image, ImagePipeline
from phenotypic.grid import ManualGridFinder
from phenotypic.measure import MeasureDinoEmbedding
from phenotypic.measure._measure_dino_embedding import GridCellOutsideImageError
from phenotypic.schema import DINO_EMBEDDING, GRID, OBJECT
from phenotypic.sdk_.exceptions_ import OperationFailedError
from tests._tiny_run import STUB_COMMIT, STUB_HIDDEN_SIZE

_HAS_TORCH = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="Requires torch")


def _tune_spec(field):
    from phenotypic.sdk_.typing_ import TuneSpec

    return next(meta for meta in field.metadata if isinstance(meta, TuneSpec))


def tune_choices(field) -> list:
    """The categorical search choices a field's ``TuneSpec`` declares."""
    return list(_tune_spec(field).categories)


def tune_bounds(field) -> tuple:
    """The ``(low, high)`` search window a field's ``TuneSpec`` declares."""
    spec = _tune_spec(field)
    return (spec.low, spec.high)


def _spy_on_crops(monkeypatch) -> list:
    """Record every crop batch the measurer hands ``embed_crops``."""
    from phenotypic.detect.nn._helper import _dino_support

    batches: list = []
    real = _dino_support.embed_crops

    def _spy(model, processor, crops, **kwargs):
        batches.append(np.array(crops))
        return real(model, processor, crops, **kwargs)

    monkeypatch.setattr(_dino_support, "embed_crops", _spy)
    return batches


# ---------------------------------------------------------------------------
# The plan's tests (P4 T3 Step 1)
# ---------------------------------------------------------------------------


def test_field_order_size_first():
    assert list(MeasureDinoEmbedding.model_fields)[:4] == ["size", "version", "revision", "regions"]


@requires_torch
def test_revision_pin_reaches_the_loader(stub_backbone, detected_plate):
    MeasureDinoEmbedding(revision="a" * 40).measure(detected_plate)
    assert stub_backbone.load_calls[-1]["revision"] == "a" * 40


def test_model_token(stub_backbone):
    assert MeasureDinoEmbedding(size="base").model_token() == "dinov2-vitb14"


@requires_torch
def test_crops_one_row_per_colony(stub_backbone, detected_plate):
    frame = MeasureDinoEmbedding().measure(detected_plate)
    assert len(frame) == detected_plate.num_objects
    assert frame["embedding"].iloc[0].dtype == np.float16


@requires_torch
def test_entry_records_the_resolved_revision(stub_backbone, detected_plate):
    m = MeasureDinoEmbedding()
    m.measure(detected_plate)
    assert m.entry_descriptor()["model"]["revision"] == "0" * 40


def test_grid_cell_outside_image_fails_the_image(stub_backbone, plate):
    m = MeasureDinoEmbedding(regions="grid",
                             grid=ManualGridFinder(row_edges=[0, 50, plate.shape[0] + 10], col_edges=[0, 80]))
    with pytest.raises(OperationFailedError, match=r"cell \(1, 0\).*image is \d+ x \d+") as excinfo:
        m.measure(plate)
    assert excinfo.value.err_type is GridCellOutsideImageError


def test_grid_edges_are_dvc_addressable():
    pipe = ImagePipeline(ops=[], meas={"emb": MeasureDinoEmbedding(regions="grid",
                         grid=ManualGridFinder(row_edges=[0, 10], col_edges=[0, 10]))})
    data = json.loads(pipe.to_json())
    assert data["meas"]["emb"]["params"]["grid"]["params"]["row_edges"] == [0, 10]
    # pins the key path the how-to page documents (spec E3a)


@requires_torch
def test_backbone_loaded_once_per_process(stub_backbone, detected_plate):
    m = MeasureDinoEmbedding()
    m.measure(detected_plate)
    m.measure(detected_plate)
    assert len(stub_backbone.load_calls) == 1


def test_dinov3_requires_license_acceptance(monkeypatch, stub_backbone, detected_plate):
    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)
    with pytest.raises(OperationFailedError, match="license") as excinfo:
        MeasureDinoEmbedding(version=3).measure(detected_plate)
    assert excinfo.value.err_type is RuntimeError
    assert len(stub_backbone.load_calls) == 0                     # refused before loading


def test_every_tunable_crop_px_fits_both_patch_sizes():
    choices = tune_choices(MeasureDinoEmbedding.model_fields["crop_px"])   # read from the TuneSpec
    assert choices == [112, 224, 336, 448]
    assert all(c % 14 == 0 and c % 16 == 0 for c in choices)

def test_crop_pad_frac_tunes_per_side_up_to_half():
    assert tune_bounds(MeasureDinoEmbedding.model_fields["crop_pad_frac"]) == (0.0, 0.5)

@requires_torch
def test_an_elongated_colony_is_not_stretched(stub_backbone, monkeypatch):
    # the crop is squared to the box's longer side before resizing
    crops = _spy_on_crops(monkeypatch)
    rgb = np.full((200, 200, 3), 0.5)
    objmap = np.zeros((200, 200), dtype=np.uint16)
    rgb[95:105, 70:130] = 1.0                       # a 10 x 60 white colony
    objmap[95:105, 70:130] = 1
    image = Image(rgb)
    image.objmap[:] = objmap
    MeasureDinoEmbedding(crop_px=112, crop_pad_frac=0.0).measure(image)
    white = crops[0][0].mean(axis=2) > 200
    rows, cols = int(white.any(axis=1).sum()), int(white.any(axis=0).sum())
    assert cols >= 110                              # the long side fills the crop
    assert 15 <= rows <= 22                         # 112 * 10 / 60 = 18.7, not 112


@requires_torch
def test_a_square_leaving_the_image_is_zero_filled_not_stretched(stub_backbone, monkeypatch):
    crops = _spy_on_crops(monkeypatch)
    rgb = np.full((200, 200, 3), 0.5)
    objmap = np.zeros((200, 200), dtype=np.uint16)
    rgb[0:20, 90:110] = 1.0                         # a colony on the top edge
    objmap[0:20, 90:110] = 1
    image = Image(rgb)
    image.objmap[:] = objmap
    MeasureDinoEmbedding(crop_px=112, crop_pad_frac=0.5).measure(image)
    crop = crops[0][0]                              # box 20 -> square 40, 10 rows above the image
    assert (crop[:25] == 0).all()                   # 112 * 10 / 40 = 28 rows of zero fill
    assert (crop[40:80, 40:70] > 200).all()         # the colony, unstretched, below it
    assert (crop[100:, :20] > 100).all() and (crop[100:, :20] < 160).all()   # grey background


# ---------------------------------------------------------------------------
# Fields, validation and identity
# ---------------------------------------------------------------------------


def test_the_fourteen_fields_in_the_plans_order():
    """#281 puts ``name`` then ``layer`` right after ``grid``."""
    assert list(MeasureDinoEmbedding.model_fields) == [
        "size", "version", "revision", "regions", "grid", "name", "layer", "pooling",
        "crop_px", "crop_pad_frac", "mask_background", "dtype", "batch_size", "device",
    ]


def test_defaults():
    m = MeasureDinoEmbedding()
    assert (m.size, m.version, m.revision, m.regions, m.grid, m.name, m.layer) == (
        "base", 2, None, "objects", None, None, "rgb"
    )
    assert (m.pooling, m.crop_px, m.crop_pad_frac, m.mask_background) == ("crop_cls", 224, 0.25, False)
    assert (m.dtype, m.batch_size, m.device) == ("float16", 64, "auto")


def test_field_order_reaches_the_json_schema_and_the_pipeline_file():
    schema = MeasureDinoEmbedding.model_json_schema()
    assert list(schema["properties"])[:7] == [
        "size", "version", "revision", "regions", "grid", "name", "layer"
    ]
    params = json.loads(MeasureDinoEmbedding().to_json())["params"]
    assert list(params)[:4] == ["size", "version", "revision", "regions"]


def test_field_descriptions_survive_the_reorder():
    assert MeasureDinoEmbedding.model_fields["revision"].description
    assert MeasureDinoEmbedding.model_fields["regions"].description
    assert MeasureDinoEmbedding.model_fields["name"].description
    assert MeasureDinoEmbedding.model_fields["layer"].description


@pytest.mark.parametrize(
    ("version", "size", "token"),
    [(2, "small", "dinov2-vits14"), (2, "large", "dinov2-vitl14"), (3, "base", "dinov3-vitb16")],
)
def test_model_token_spells_version_size_and_patch(version, size, token):
    m = MeasureDinoEmbedding(version=version, size=size)
    assert m.model_token() == token
    assert m.payload_key() == f"{token}/rgb/objects"


def test_a_grid_embedder_writes_the_grid_folder():
    m = MeasureDinoEmbedding(regions="grid", grid=ManualGridFinder(row_edges=[0, 10], col_edges=[0, 10]))
    assert m.payload_key() == "dinov2-vitb14/rgb/grid"
    assert m.requires_objects is False


def test_a_named_embedder_writes_its_named_folder():
    """#281 named regions: ``grid-<name>`` and ``objects-<name>``."""
    m = MeasureDinoEmbedding(
        regions="grid", grid=ManualGridFinder(row_edges=[0, 10], col_edges=[0, 10]),
        name="wells",
    )
    assert m.payload_key() == "dinov2-vitb14/rgb/grid-wells"
    assert m.requires_objects is False
    crops = MeasureDinoEmbedding(name="cls")
    assert crops.payload_key() == "dinov2-vitb14/rgb/objects-cls"
    assert crops.requires_objects is True
    with pytest.raises(ValueError, match="name must be a lowercase slug"):
        MeasureDinoEmbedding(
            regions="grid", grid=ManualGridFinder(row_edges=[0, 10], col_edges=[0, 10]),
            name="Wells",
        )


@requires_torch
def test_two_named_grids_share_one_backbone_load(stub_backbone, plate):
    """rig1's ``wells`` and ``agar`` embedders of one model load it once, and
    each payload holds its own grid's cells."""
    from tests._tiny_run import grid_embedder, named_grid_embedders

    payloads = ImagePipeline(meas=named_grid_embedders(grid_embedder)).embed(plate)
    assert list(payloads) == ["dinov2-vitb14/rgb/grid-wells", "dinov2-vitb14/rgb/grid-agar"]
    assert [len(p.frame) for p in payloads.values()] == [4, 1]
    assert len(stub_backbone.load_calls) == 1
    wells, agar = (p.frame["embedding"] for p in payloads.values())
    assert not any(np.array_equal(agar[0], cell) for cell in wells)


def _equal_channel_plate(plate):
    """The synth plate with its three channels made equal: gray is then rgb."""
    value = np.asarray(plate.rgb[:]).mean(axis=2).astype(np.uint8)
    return Image(np.stack([value] * 3, axis=-1))


@requires_torch
@pytest.mark.parametrize("pooling", ["crop_cls", "masked_dense"])
def test_a_gray_dino_grid_differs_from_rgb_and_equals_it_on_a_gray_plate(stub_backbone, plate, pooling):
    """#281 layer: gray is the gray layer as three uint8 channels, so on a
    plate whose channels are already equal it is the rgb embedding."""
    from tests._tiny_run import grid_embedder

    rgb = grid_embedder(pooling=pooling)
    gray = grid_embedder(pooling=pooling, layer="gray")
    coloured = [np.stack(e.measure(plate)["embedding"]).astype(np.float32) for e in (rgb, gray)]
    assert not np.allclose(coloured[0], coloured[1])
    equal = _equal_channel_plate(plate)
    same = [np.stack(e.measure(equal)["embedding"]).astype(np.float32) for e in (rgb, gray)]
    np.testing.assert_allclose(same[0], same[1], atol=1e-2)
    assert len(stub_backbone.load_calls) == 1


@requires_torch
@pytest.mark.parametrize("kwargs", [{"mask_background": True}, {"pooling": "masked_dense"}])
def test_a_gray_object_embedder_runs_every_crop_option(stub_backbone, detected_plate, kwargs):
    rgb = MeasureDinoEmbedding(device="cpu", **kwargs).measure(detected_plate)
    gray = MeasureDinoEmbedding(device="cpu", layer="gray", **kwargs).measure(detected_plate)
    assert len(gray) == len(rgb) == detected_plate.num_objects
    assert not np.allclose(np.stack(rgb["embedding"]), np.stack(gray["embedding"]))


@requires_torch
def test_a_dino_entry_records_its_layer(stub_backbone, plate):
    from tests._tiny_run import grid_embedder

    payload = grid_embedder(layer="gray", name="wells").payload(plate)
    assert payload.key == "dinov2-vitb14/gray/grid-wells"
    assert payload.descriptor["input"] == {"series": "rgb", "level": 0, "layer": "gray"}


@pytest.mark.parametrize("kwargs", [{"mask_background": True}, {"crop_pad_frac": 0.5}])
def test_objects_only_crop_options_are_refused_for_a_grid(kwargs):
    with pytest.raises(ValueError, match="objects only"):
        MeasureDinoEmbedding(
            regions="grid", grid=ManualGridFinder(row_edges=[0, 10], col_edges=[0, 10]), **kwargs
        )


@pytest.mark.parametrize(
    "kwargs",
    [{"crop_px": 0}, {"crop_px": -14}, {"crop_pad_frac": -0.1}, {"batch_size": 0},
     {"revision": ""}, {"revision": "  "}, {"dtype": "float32"}, {"pooling": "cls"},
     {"device": "mps"}],
)
def test_invalid_values_are_refused(kwargs):
    with pytest.raises(ValueError):
        MeasureDinoEmbedding(**kwargs)


def test_the_pipeline_file_round_trips():
    m = MeasureDinoEmbedding(size="small", revision="b" * 40, regions="grid",
                             grid=ManualGridFinder(row_edges=[0, 50], col_edges=[0, 80]),
                             pooling="crop_mean", crop_px=112)
    pipe = ImagePipeline(ops=[], meas={"emb": m})
    rebuilt = ImagePipeline.from_json(pipe.to_json())
    again = rebuilt._meas["emb"]
    assert isinstance(again, MeasureDinoEmbedding)
    assert again.model_dump(mode="json") == m.model_dump(mode="json")


def test_tune_annotations_follow_the_plan():
    from phenotypic.sdk_.typing_ import TuneSpec

    def spec(name):
        return next(meta for meta in MeasureDinoEmbedding.model_fields[name].metadata
                    if isinstance(meta, TuneSpec))

    for name in ("size", "version", "revision", "regions", "grid", "name", "layer", "dtype",
                 "batch_size", "device"):
        assert spec(name).tunable is False, name
    assert spec("crop_px").categories == (112, 224, 336, 448)
    assert (spec("crop_pad_frac").low, spec("crop_pad_frac").high) == (0.0, 0.5)


def test_measurement_info_is_the_embedding_schema():
    assert MeasureDinoEmbedding._measurement_infoclass is DINO_EMBEDDING
    assert [member.name for member in DINO_EMBEDDING] == ["EMBEDDING"]
    assert DINO_EMBEDDING.EMBEDDING.bio_desc == ""


def test_preflight_declares_torch_transformers_and_the_weights():
    req = MeasureDinoEmbedding(size="small").preflight_requirements()
    assert set(req.modules) == {"torch", "transformers"}
    assert req.extra == "foundation"
    assert req.rgb_input is True
    assert [w.model for w in req.weights] == ["facebook/dinov2-small"]
    assert MeasureDinoEmbedding(layer="gray").preflight_requirements().rgb_input is False
    gated = MeasureDinoEmbedding(version=3).preflight_requirements()
    assert "huggingface_hub" in gated.modules
    assert gated.weights[0].license_key == "dinov3"


# ---------------------------------------------------------------------------
# Embedding behaviour on the stub
# ---------------------------------------------------------------------------


@requires_torch
@pytest.mark.parametrize("pooling", ["crop_cls", "crop_mean", "masked_dense"])
def test_every_pooling_gives_one_distinct_vector_per_colony(stub_backbone, detected_plate, pooling):
    frame = MeasureDinoEmbedding(pooling=pooling, crop_px=112).measure(detected_plate)
    assert list(frame.columns) == [str(OBJECT.LABEL), "embedding"]
    assert sorted(frame[str(OBJECT.LABEL)]) == detected_plate.objects.labels
    vectors = np.stack(frame["embedding"].to_list())
    assert vectors.shape == (detected_plate.num_objects, STUB_HIDDEN_SIZE)
    assert vectors.dtype == np.float16
    assert np.isfinite(vectors).all()
    assert len(np.unique(vectors, axis=0)) > 1


@requires_torch
def test_vectors_do_not_depend_on_the_batch_size(stub_backbone, detected_plate):
    one = MeasureDinoEmbedding(batch_size=1, crop_px=112).measure(detected_plate)
    many = MeasureDinoEmbedding(batch_size=500, crop_px=112).measure(detected_plate)
    np.testing.assert_array_equal(np.stack(one["embedding"]), np.stack(many["embedding"]))


@requires_torch
def test_each_pooling_reads_its_own_tokens(stub_backbone, detected_plate):
    """``crop_cls`` is the CLS, ``crop_mean`` the patch mean, ``masked_dense``
    the plate's patches under the mask: three different vectors per colony.

    The stub's CLS is the patches' maximum, not their mean (review T2), so a
    measurer that ignored ``pooling`` (review T1's first mutant) fails here.
    The patch mean is checked against the crop's tokens directly.
    """
    from phenotypic.detect.nn._helper._dino_support import embed_crops
    from tests._tiny_run import _build_stub_dino

    vectors = {
        pooling: np.stack(
            MeasureDinoEmbedding(pooling=pooling, crop_px=112)
            .measure(detected_plate)["embedding"]
        ).astype(np.float32)
        for pooling in ("crop_cls", "crop_mean", "masked_dense")
    }
    for a, b in (("crop_cls", "crop_mean"), ("crop_cls", "masked_dense"),
                 ("crop_mean", "masked_dense")):
        assert np.abs(vectors[a] - vectors[b]).max() > 1e-2, (a, b)
    # Per colony, the CLS (max) bounds the mean from above, component-wise.
    assert (vectors["crop_cls"] >= vectors["crop_mean"] - 1e-2).all()
    model, processor = _build_stub_dino()
    crops = np.random.default_rng(1).integers(0, 256, (2, 112, 112, 3), dtype=np.uint8)
    mean = embed_crops(model, processor, crops, pooling="crop_mean", device="cpu", batch_size=2)
    cls = embed_crops(model, processor, crops, pooling="crop_cls", device="cpu", batch_size=2)
    assert not np.allclose(mean, cls)


@requires_torch
@pytest.mark.parametrize("pooling", ["crop_mean", "masked_dense"])
def test_register_tokens_never_reach_a_pooled_vector(monkeypatch, detected_plate, pooling):
    """A backbone with four register tokens of 1000 gives the vectors of one
    with none: ``crop_mean`` and ``masked_dense`` drop them (review T3)."""
    from phenotypic.measure import _measure_dino_embedding as module
    from tests._tiny_run import install_stub_backbone

    def embed() -> np.ndarray:
        frame = MeasureDinoEmbedding(pooling=pooling, crop_px=112).measure(detected_plate)
        return np.stack(frame["embedding"]).astype(np.float32)

    install_stub_backbone(monkeypatch)
    plain = embed()
    install_stub_backbone(monkeypatch, num_register_tokens=4)
    with_registers = embed()
    module._cached_backbone.cache_clear()
    assert np.abs(with_registers).max() < 100          # no 1000 leaked in
    np.testing.assert_array_equal(with_registers, plain)


@requires_torch
def test_masked_dense_pools_the_colony_mask_not_its_box(stub_backbone):
    """A colony shaped like an L leaves one patch of its bounding box outside
    its mask. Changing that patch's pixels must not move the colony's vector
    (review T1's second mutant pools the whole box)."""
    patch = 14

    def embed(corner_value: float) -> np.ndarray:
        rgb = np.full((112, 112, 3), 0.3)
        objmap = np.zeros((112, 112), dtype=np.uint16)
        objmap[patch:3 * patch, patch:3 * patch] = 1          # a 2 x 2-patch square
        objmap[2 * patch:3 * patch, 2 * patch:3 * patch] = 0  # minus one patch: an L
        rgb[objmap == 1] = 0.9
        rgb[2 * patch:3 * patch, 2 * patch:3 * patch] = corner_value
        image = Image(rgb)
        image.objmap[:] = objmap
        frame = MeasureDinoEmbedding(pooling="masked_dense", crop_px=112).measure(image)
        return np.stack(frame["embedding"]).astype(np.float32)

    np.testing.assert_array_equal(embed(0.3), embed(0.0))


@requires_torch
def test_a_grid_cell_is_cropped_exactly_not_padded_or_squared(stub_backbone, monkeypatch):
    """A 100 x 50 white cell on grey becomes an all-white crop: no padding
    and no squaring, which would pull grey in (review T1's third mutant)."""
    crops = _spy_on_crops(monkeypatch)
    rgb = np.full((300, 300, 3), 0.5)
    rgb[50:150, 100:150] = 1.0
    grid = ManualGridFinder(row_edges=[50, 150], col_edges=[100, 150])
    MeasureDinoEmbedding(regions="grid", grid=grid, crop_px=112).measure(Image(rgb))
    (crop,) = crops[0]
    assert crop.shape == (112, 112, 3)
    assert (crop >= 250).all()


@requires_torch
def test_crop_mean_leaves_out_the_cls_and_register_tokens():
    """A backbone whose CLS and two register tokens are 1000: ``crop_cls``
    reads the CLS, and ``crop_mean`` none of the three."""
    from types import SimpleNamespace

    import torch

    from phenotypic.detect.nn._helper._dino_support import embed_crops
    from tests._tiny_run import _build_stub_dino

    stub, processor = _build_stub_dino()

    class _LoudTokens(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(**{**vars(stub.config), "num_register_tokens": 2})

        def forward(self, pixel_values):
            tokens = stub(pixel_values).last_hidden_state
            loud = torch.full_like(tokens[:, :3], 1000.0)
            return SimpleNamespace(last_hidden_state=torch.cat([loud, tokens[:, 1:]], dim=1))

    crops = np.random.default_rng(0).integers(0, 256, (3, 28, 28, 3), dtype=np.uint8)
    cls = embed_crops(_LoudTokens(), processor, crops, pooling="crop_cls", device="cpu", batch_size=2)
    mean = embed_crops(_LoudTokens(), processor, crops, pooling="crop_mean", device="cpu", batch_size=2)
    plain = embed_crops(stub, processor, crops, pooling="crop_mean", device="cpu", batch_size=2)
    assert (cls == 1000).all()
    np.testing.assert_allclose(mean, plain, rtol=1e-6)


@requires_torch
def test_mask_background_and_padding_change_the_crop(stub_backbone, detected_plate):
    base = np.stack(MeasureDinoEmbedding(crop_px=112).measure(detected_plate)["embedding"])
    masked = np.stack(MeasureDinoEmbedding(crop_px=112, mask_background=True)
                      .measure(detected_plate)["embedding"])
    tight = np.stack(MeasureDinoEmbedding(crop_px=112, crop_pad_frac=0.0)
                     .measure(detected_plate)["embedding"])
    assert not np.array_equal(base, masked)
    assert not np.array_equal(base, tight)


@requires_torch
def test_a_grid_embeds_every_cell_without_objects(stub_backbone, plate):
    grid = ManualGridFinder(row_edges=[0, 100, 200], col_edges=[0, 150, 300])
    frame = MeasureDinoEmbedding(regions="grid", grid=grid, crop_px=112).measure(plate)
    assert list(frame[str(GRID.ROW_MAJOR_IDX)]) == [0, 1, 2, 3]
    assert np.stack(frame["embedding"]).shape == (4, STUB_HIDDEN_SIZE)


@requires_torch
def test_masked_dense_grid_cells_pool_their_own_patches(stub_backbone, plate):
    grid = ManualGridFinder(row_edges=[0, 140, 280], col_edges=[0, 140, 280])
    frame = MeasureDinoEmbedding(regions="grid", grid=grid, pooling="masked_dense",
                                 crop_px=112).measure(plate)
    vectors = np.stack(frame["embedding"])
    assert vectors.shape == (4, STUB_HIDDEN_SIZE)
    assert len(np.unique(vectors, axis=0)) == 4


@requires_torch
def test_masked_dense_a_region_smaller_than_a_patch_still_embeds(stub_backbone, plate):
    """A 3 x 3 cell rounds to no patch under nearest sampling; it falls back
    to the patch it touches rather than to a zero vector."""
    grid = ManualGridFinder(row_edges=[20, 23], col_edges=[20, 23])
    frame = MeasureDinoEmbedding(regions="grid", grid=grid, pooling="masked_dense",
                                 crop_px=112).measure(plate)
    assert np.abs(frame["embedding"].iloc[0]).sum() > 0


@requires_torch
def test_crop_px_must_be_a_multiple_of_the_patch_size(stub_backbone, detected_plate):
    with pytest.raises(OperationFailedError, match="multiple of the patch size 14") as excinfo:
        MeasureDinoEmbedding(crop_px=100).measure(detected_plate)
    assert excinfo.value.err_type is ValueError


@requires_torch
def test_the_documented_header_is_the_column_written(stub_backbone, detected_plate):
    """The Measurements reference's column for DINO_EMBEDDING.EMBEDDING is the
    table's own ``embedding`` column (the generic check in
    ``tests/unit/util/test_output_headers.py`` skips embedders: it would load
    real weights)."""
    from phenotypic.util import measurement_producers

    producer = next(p for p in measurement_producers() if p.producer is MeasureDinoEmbedding)
    documented = {producer.output_header(member) for info in producer.primary_infos for member in info}
    assert documented == {"embedding"}
    assert documented <= set(MeasureDinoEmbedding(crop_px=112).measure(detected_plate).columns)


@requires_torch
def test_no_objects_gives_an_empty_table(stub_backbone, plate):
    plate.objmap[:] = 0
    frame = MeasureDinoEmbedding().measure(plate)
    assert list(frame.columns) == [str(OBJECT.LABEL), "embedding"]
    assert len(frame) == 0


@requires_torch
def test_the_pipeline_payload_carries_the_model_entry(stub_backbone, detected_plate):
    pipe = ImagePipeline(
        ops=[], meas={"emb": MeasureDinoEmbedding(size="small", crop_px=112, device="cpu")}
    )
    payloads = pipe.embed(detected_plate)
    assert list(payloads) == ["dinov2-vits14/rgb/objects"]
    descriptor = payloads["dinov2-vits14/rgb/objects"].descriptor
    assert descriptor["model"] == {
        "family": "dinov2", "hf_id": "facebook/dinov2-small",
        "revision": STUB_COMMIT, "patch_size": 14, "device": "cpu", "device_name": None,
    }
    assert (descriptor["dim"], descriptor["dtype"], descriptor["pooling"]) == (
        STUB_HIDDEN_SIZE, "float16", "crop_cls")
    assert descriptor["producer"]["class"] == "MeasureDinoEmbedding"


@requires_torch
@pytest.mark.parametrize("cuda", [False, True])
def test_an_auto_embedder_records_the_device_that_ran(stub_backbone, monkeypatch, cuda):
    """#281 P4 review Q1 follow-up: ``device="auto"`` is the same field value
    on a GPU node and a CPU node, so the entry records the device
    ``resolve_backbone_device`` resolved, the one the backbone was loaded on.
    No forward pass runs, so the stub is never moved to a GPU."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    # The entry also names the GPU; this node may have none to ask.
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "stub GPU")
    m = MeasureDinoEmbedding(size="small", device="auto")
    expected = "cuda" if cuda else "cpu"
    assert m.entry_descriptor()["model"]["device"] == expected
    assert stub_backbone.load_calls[-1]["device"] == expected


@requires_torch
def test_a_cuda_entry_records_the_gpu_name(stub_backbone, monkeypatch):
    """#281: the device is provenance. On CUDA the entry also names the GPU
    model (``torch.cuda.get_device_name()``); on the CPU ``device_name`` is
    ``None``. No forward pass runs, so the stub is never moved to a GPU."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "NVIDIA RTX 6000 Ada Generation")
    model = MeasureDinoEmbedding(size="small", device="auto").entry_descriptor()["model"]
    assert (model["device"], model["device_name"]) == ("cuda", "NVIDIA RTX 6000 Ada Generation")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = MeasureDinoEmbedding(size="small", device="auto").entry_descriptor()["model"]
    assert (model["device"], model["device_name"]) == ("cpu", None)


@requires_torch
def test_embed_crops_refuses_a_resized_or_mistyped_batch(stub_backbone):
    from phenotypic.detect.nn._helper._dino_support import embed_crops
    from tests._tiny_run import _build_stub_dino

    model, processor = _build_stub_dino()
    crops = np.zeros((2, 28, 28, 3), np.uint8)
    out = embed_crops(model, processor, crops, pooling="crop_mean", device="cpu", batch_size=1)
    assert out.shape == (2, STUB_HIDDEN_SIZE) and out.dtype == np.float32
    with pytest.raises(ValueError, match="uint8"):
        embed_crops(model, processor, crops.astype(np.float32), pooling="crop_cls",
                    device="cpu", batch_size=1)
    with pytest.raises(ValueError, match="pooling"):
        embed_crops(model, processor, crops, pooling="masked_dense", device="cpu", batch_size=1)


# ---------------------------------------------------------------------------
# Real-model smoke test (P4 T3 Step 5): Slurm only, never in PR runs
# ---------------------------------------------------------------------------


def _dinov2_small_is_available() -> bool:
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        return False
    from phenotypic.detect.nn._helper._checkpoint_manager import hf_repo_is_cached

    return bool(hf_repo_is_cached("facebook/dinov2-small"))


@pytest.mark.slow
@pytest.mark.skipif(not _dinov2_small_is_available(),
                    reason="Requires torch, transformers and cached facebook/dinov2-small weights")
def test_real_dinov2_small_embeds_the_synth_plate_on_cpu(detected_plate, capsys):
    """DINOv2-small on the synth plate, CPU, real weights (P4 T3 Step 5).

    Cached weights are not enough to run offline: transformers 5.12's
    ``AutoImageProcessor`` probes ``processor_config.json`` before
    ``preprocessor_config.json``, and dinov2-small has only the latter, so a
    cached snapshot still asks the hub. A missing file is tolerated, but an
    expired stored Hugging Face token turns the probe into a 401
    (``RepositoryNotFoundError``) and the load fails. Run it with a valid
    token, or anonymously (``HF_HUB_DISABLE_IMPLICIT_TOKEN=1``) for this
    public repo.
    """
    from phenotypic.measure import _measure_dino_embedding as module

    module._cached_backbone.cache_clear()
    m = MeasureDinoEmbedding(size="small", device="cpu")
    start = time.perf_counter()
    frame = m.measure(detected_plate)
    elapsed = time.perf_counter() - start
    descriptor = m.entry_descriptor()
    vectors = np.stack(frame["embedding"]).astype(np.float32)
    with capsys.disabled():
        print(f"\nSMOKE objects={len(frame)} dim={vectors.shape[1]} "
              f"measure_seconds={elapsed:.2f} revision={descriptor['model']['revision']}")
    assert len(frame) == detected_plate.num_objects
    assert vectors.shape[1] == descriptor["dim"] == 384
    assert np.isfinite(vectors).all()
    assert descriptor["model"]["hf_id"] == "facebook/dinov2-small"
    assert descriptor["model"]["patch_size"] == 14
    revision = descriptor["model"]["revision"]
    assert isinstance(revision, str) and len(revision) == 40
    # Distinct colonies embed to distinct directions.
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    assert (unit @ unit.T)[np.triu_indices(len(unit), 1)].min() < 0.999
    module._cached_backbone.cache_clear()
