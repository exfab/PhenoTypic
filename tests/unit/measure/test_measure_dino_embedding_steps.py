"""``MeasureDinoEmbedding`` on the three steps (#282 P1 T2). [torch]

``prepare_regions`` cuts every crop (or, for ``masked_dense``, every
processor-normalised tile) on the CPU with no model, so #282's Stage 4 can
run it in DataLoader workers; the inline path is the same preparation with
``_InlineDinoEncoder`` in the middle. The inline bytes are pinned by
``test_embedding_inline_golden.py``. Torch tests skip without torch; the
torch-free probe runs everywhere.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from phenotypic.measure import MeasureDinoEmbedding

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="Requires torch"
)


def _empty(plate):
    """*plate* with no objects: a copy whose objmap is cleared."""
    empty = plate.copy()
    empty.objmap[:] = 0
    return empty


@requires_torch
def test_prepare_loads_no_model(stub_backbone, detected_plate):
    MeasureDinoEmbedding().prepare_regions(detected_plate)
    assert stub_backbone.load_calls == []


def test_prepare_imports_no_torch():
    """Crop preparation imports no torch: a fresh process is the probe."""
    code = (
        "import sys; from phenotypic.measure import MeasureDinoEmbedding;"
        "from phenotypic.data import load_synth_yeast_plate;"
        "from phenotypic.detect import OtsuDetector;"
        "p = OtsuDetector().apply(load_synth_yeast_plate());"
        "MeasureDinoEmbedding().prepare_regions(p); print('torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


@requires_torch
def test_crops_are_the_inline_crops(stub_backbone, detected_plate):
    m = MeasureDinoEmbedding(crop_pad_frac=0.5, mask_background=True)
    prepared = m.prepare_regions(detected_plate)
    assert prepared.inputs.kind == "crops"
    assert prepared.inputs.pooling == "crop_cls"
    assert prepared.inputs.arrays[0].shape == (detected_plate.num_objects, 224, 224, 3)
    assert prepared.inputs.arrays[0].dtype == np.uint8
    assert m.crop_geometry() == (224, 224)


@requires_torch
def test_dense_tiles_are_the_processors_pixels(stub_backbone, detected_plate):
    from phenotypic.detect.nn._helper._dino_support import NATIVE_PROCESSOR_KWARGS
    from phenotypic.measure import _measure_dino_embedding as module

    m = MeasureDinoEmbedding(pooling="masked_dense", crop_px=112)
    _, processor = module.load_dino_backbone(2, "base", "cpu")  # the stub's
    prepared = m.prepare_regions(detected_plate, processor=processor)
    assert prepared.inputs.kind == "tiles" and prepared.inputs.pooling == "patches"
    assert all(
        t.dtype == np.float32 and t.shape[1] % 14 == 0 and t.shape[2] % 14 == 0
        for t in prepared.inputs.arrays
    )
    first = np.asarray(m._layer_pixels(detected_plate))[:112, :112]
    # the inline path: extract_patch_features asks for "pt" (_dino_support.py)
    ref = processor(images=first, return_tensors="pt", **NATIVE_PROCESSOR_KWARGS)[
        "pixel_values"
    ][0].numpy()
    assert np.array_equal(prepared.inputs.arrays[0], ref)  # same bits as inline


@requires_torch
def test_zero_regions_trigger_no_second_load(stub_backbone, plate):
    m = MeasureDinoEmbedding()
    vectors = m.encode(m.prepare_regions(_empty(plate)), object())  # never .run
    assert vectors.shape == (0, 0)
    assert stub_backbone.load_calls == []


@requires_torch
def test_entry_descriptor_for_does_not_load(stub_backbone, detected_plate):
    from phenotypic.measure._measure_dino_embedding import _cached_backbone

    m = MeasureDinoEmbedding(device="cpu")
    m.measure(detected_plate)
    model = _cached_backbone(2, "base", None, "cpu", "float32")[0]
    n = len(stub_backbone.load_calls)
    assert m.entry_descriptor_for(model) == m.entry_descriptor()
    assert len(stub_backbone.load_calls) == n


@requires_torch
def test_the_inline_encoder_feeds_one_batch_at_a_time(stub_backbone, detected_plate, monkeypatch):
    """``prepare_regions`` builds every crop, but the inline encoder hands
    ``embed_crops`` one ``batch_size`` slice per call, so peak device memory is
    as it was before the refactor."""
    from phenotypic.detect.nn._helper import _dino_support

    sizes: list[int] = []
    real = _dino_support.embed_crops

    def _spy(model, processor, crops, **kwargs):
        sizes.append(len(crops))
        return real(model, processor, crops, **kwargs)

    monkeypatch.setattr(_dino_support, "embed_crops", _spy)
    n = detected_plate.num_objects
    MeasureDinoEmbedding(crop_px=112, batch_size=10, device="cpu").measure(detected_plate)
    assert sizes == [10] * (n // 10) + ([n % 10] if n % 10 else [])


@requires_torch
def test_the_inline_path_cuts_crops_one_batch_at_a_time(stub_backbone, detected_plate, monkeypatch):
    """#282 review I4: on the inline path, host memory holds one batch of crops, not the
    image's. When ``embed_crops`` receives its k-th slice, exactly k * batch_size crops
    (all n at the last) have been cut."""
    from phenotypic.detect.nn._helper import _dino_support

    cut: list[int] = []
    at_each_batch: list[int] = []
    real_crop, real_embed = MeasureDinoEmbedding._crop, _dino_support.embed_crops

    def _counting_crop(self, *args):
        cut.append(1)
        return real_crop(self, *args)

    def _spy(model, processor, crops, **kwargs):
        at_each_batch.append(len(cut))
        return real_embed(model, processor, crops, **kwargs)

    monkeypatch.setattr(MeasureDinoEmbedding, "_crop", _counting_crop)
    monkeypatch.setattr(_dino_support, "embed_crops", _spy)
    n = detected_plate.num_objects
    assert n > 30
    MeasureDinoEmbedding(crop_px=112, batch_size=10, device="cpu").payload(detected_plate)
    assert at_each_batch == [min(10 * (k + 1), n) for k in range(-(-n // 10))]


@requires_torch
@pytest.mark.parametrize("pooling", ["crop_cls", "crop_mean"])
def test_deferred_crops_are_the_materialised_crops(stub_backbone, detected_plate, pooling):
    """The inline path's deferred crops are the same bytes, so its vectors are too."""
    m = MeasureDinoEmbedding(crop_px=112, batch_size=10, device="cpu", pooling=pooling,
                             mask_background=True)
    whole = m.prepare_regions(detected_plate)
    deferred = m.prepare_regions(detected_plate, materialise=False)
    (array,), (lazy,) = whole.inputs.arrays, deferred.inputs.arrays
    assert not isinstance(lazy, np.ndarray) and lazy.shape == array.shape and len(lazy) == len(array)
    assert np.array_equal(np.asarray(lazy), array) and np.array_equal(lazy[3:17], array[3:17])
    encoder = m.inline_encoder(detected_plate)
    assert np.array_equal(m.encode(whole, encoder), m.encode(deferred, encoder))


@requires_torch
def test_inline_dense_tiles_use_the_backbones_own_processor(stub_backbone, detected_plate, monkeypatch):
    """#282 review M11: the inline ``masked_dense`` path loads no second processor."""
    from phenotypic.measure import _measure_dino_embedding as module

    loads: list[tuple] = []
    real = module.load_dino_processor

    def _counting(*args, **kwargs):
        loads.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "load_dino_processor", _counting)
    MeasureDinoEmbedding(pooling="masked_dense", crop_px=112, device="cpu").payload(detected_plate)
    assert loads == []
    MeasureDinoEmbedding(pooling="masked_dense", crop_px=112).prepare_regions(detected_plate)
    assert len(loads) == 1                            # prepare_regions alone still loads one


@requires_torch
def test_dense_vectors_pool_the_encoders_tiles(stub_backbone, detected_plate):
    """``vectors_from`` assembles the tiles the encoder returns: a swapped-in
    encoder that returns the inline encoder's grids gives the inline vectors."""
    m = MeasureDinoEmbedding(pooling="masked_dense", crop_px=112, device="cpu")
    prepared = m.prepare_regions(detected_plate)
    inline = m.encode(prepared, m.inline_encoder(detected_plate))
    grids = m.inline_encoder(detected_plate).run(prepared.inputs)
    assert len(grids) == len(prepared.inputs.arrays)

    class _Replay:
        def run(self, inputs):
            return [g.copy() for g in grids]

    np.testing.assert_array_equal(m.encode(prepared, _Replay()), inline)
    assert inline.shape == (detected_plate.num_objects, grids[0].shape[-1])
