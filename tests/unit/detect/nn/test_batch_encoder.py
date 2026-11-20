"""``BatchEncoder``: Stage 4's fixed-shape, cross-image backbone runner (#282 P2 T3). [torch]

CPU only, on P0 T2's parametric stub (``_build_param_stub_dino``), whose rows
never see their batch neighbours, and its ``leaky=True`` variant, whose rows
do, so the isolation check is shown able to fail (plan review I13).

**References run at the encoder's physical batch.** The CPU's BLAS (MKL here)
picks its GEMM kernel by the row count, so a row's float32 bits can differ
between a pass of 5 rows and a pass of 8 (measured on the stub's ``Linear``:
5, 10, 11, 13, ... of 64 differ), while at one fixed pass size a row's bits
do not depend on its neighbours or its position. That is spec goal 3 as
restated (S1): invariance within one physical batch. So :func:`embed_alone`
runs the rows alone, zero-padded to ``m``, through the inline
``embed_crops``, not through the encoder.
"""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic.abc_ import BackboneInputs
from phenotypic.detect.nn._helper._batch_encoder import (
    MAX_IN_FLIGHT_OUTPUT_BYTES,
    BatchEncoder,
    Stage4OutOfMemory,
)
from phenotypic.measure import MeasureDinoEmbedding
from tests._tiny_run import STUB_HIDDEN_SIZE, STUB_PATCH_SIZE, _build_param_stub_dino

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Module-local helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_model():
    return _build_param_stub_dino()


@pytest.fixture
def leaky_stub_model():
    return _build_param_stub_dino(leaky=True)


def crops(n, *, px=224, seed=0, pooling="crop_cls"):
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(n, px, px, 3), dtype=np.uint8)
    return BackboneInputs(kind="crops", arrays=(array,), pooling=pooling)


def tiles(n, *, px, seed=0):
    rng = np.random.default_rng(seed)
    arrays = tuple(rng.normal(size=(3, px, px)).astype(np.float32) for _ in range(n))
    return BackboneInputs(kind="tiles", arrays=arrays, pooling="patches")


def tile_output_bytes(model_pair, *, px):
    """One ``px`` tile's ``(px // p, px // p, D)`` float32 grid, in bytes."""
    model, _ = model_pair
    return (px // model.config.patch_size) ** 2 * model.config.hidden_size * 4


def _record(model_pair, what):
    model, _ = model_pair
    seen = []
    model.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(what(kwargs["pixel_values"])), with_kwargs=True
    )
    return seen


def record_batch_sizes(model_pair):
    return _record(model_pair, lambda pixels: int(pixels.shape[0]))


def record_batch_shapes(model_pair):
    return _record(model_pair, lambda pixels: tuple(pixels.shape))


def embed_alone(model_pair, inputs, m):
    """*inputs*' crops alone, zero-padded to passes of exactly *m*, by the inline ``embed_crops``."""
    from phenotypic.detect.nn._helper._dino_support import embed_crops

    (array,) = inputs.arrays
    n = len(array)
    padded = np.zeros((-(-n // m) * m, *array.shape[1:]), dtype=np.uint8)
    padded[:n] = array
    model, processor = model_pair
    return embed_crops(model, processor, padded, pooling=inputs.pooling,
                       device="cpu", batch_size=m)[:n]


def _raising_oom(times, *, on_call=None):
    """A fresh stub whose first *times* passes (every pass: ``None``), or only pass *on_call*, run out."""
    model, processor = _build_param_stub_dino()
    calls = []

    def hook(module, args, kwargs):
        calls.append(int(kwargs["pixel_values"].shape[0]))
        failing = len(calls) == on_call if on_call is not None else times is None or len(calls) <= times
        if failing:
            raise torch.OutOfMemoryError("CUDA out of memory. (test)")

    model.register_forward_pre_hook(hook, with_kwargs=True)
    return model, processor, calls


def oom_once(_model_pair):
    """A fresh stub (same fixed-seed weights) whose first forward pass runs out of memory."""
    return _raising_oom(1)[:2]


def oom_always(_model_pair):
    """A fresh stub whose every forward pass runs out of memory."""
    return _raising_oom(None)[:2]


def encoder(model_pair, m, **kwargs):
    return BatchEncoder(*model_pair, device="cpu", compute_dtype="float32", physical_batch=m, **kwargs)


# ---------------------------------------------------------------------------
# The plan's tests (P2 T3 Step 1)
# ---------------------------------------------------------------------------


def test_every_forward_pass_has_exactly_m_inputs(stub_model):
    seen = record_batch_sizes(stub_model)
    enc = BatchEncoder(*stub_model, device="cpu", compute_dtype="float32", physical_batch=8)
    enc.submit("a", crops(5))
    enc.submit("b", crops(6))
    enc.flush()
    assert seen == [8, 8]


def test_vectors_route_back_by_ticket_and_row(stub_model):
    enc = BatchEncoder(*stub_model, device="cpu", compute_dtype="float32", physical_batch=4)
    a, b = crops(3, seed=1), crops(5, seed=2)
    enc.submit("a", a)
    enc.submit("b", b)
    enc.flush()
    out = dict(enc.drain())
    assert np.array_equal(out["a"][0], embed_alone(stub_model, a, 4))
    assert np.array_equal(out["b"][0], embed_alone(stub_model, b, 4))
    assert not np.array_equal(out["a"][0], out["b"][0][:3])       # the two are told apart


def rows_are_isolated(model_pair) -> bool:
    probe = crops(1, seed=7)
    vectors = []
    for seed in (1, 2, 3):
        enc = BatchEncoder(*model_pair, device="cpu", compute_dtype="float32", physical_batch=4)
        enc.submit("n", crops(seed + 1, seed=seed))
        enc.submit("p", probe)
        enc.flush()
        vectors.append(dict(enc.drain())["p"][0])
    return all(np.array_equal(v, vectors[0]) for v in vectors)


def test_a_row_does_not_depend_on_its_neighbours_on_cpu(stub_model):
    assert rows_are_isolated(stub_model)


def test_the_isolation_check_catches_a_leaky_model(leaky_stub_model):
    assert not rows_are_isolated(leaky_stub_model)              # the check above can fail (plan review I13)


def test_a_finished_image_is_drained_before_a_partial_batch(stub_model):
    enc = BatchEncoder(*stub_model, device="cpu", compute_dtype="float32", physical_batch=4)
    enc.submit("full", crops(4))
    enc.submit("partial", crops(1))
    assert [t for t, _ in enc.drain()] == ["full"]              # not held behind "partial"


def test_crop_pipelines_keep_full_batches(stub_model):
    seen = record_batch_sizes(stub_model)
    enc = BatchEncoder(*stub_model, device="cpu", compute_dtype="float32", physical_batch=8)
    for i in range(20):                                          # 20 images of 3 crops: 60 rows
        enc.submit(i, crops(3, seed=i))
    enc.flush()
    assert seen[:-1] == [8] * 7 and len(seen) == 8               # no early flush; only the last is padded


def test_dense_grids_flush_on_the_byte_bound(stub_model):
    seen = record_batch_sizes(stub_model)
    one_grid = tile_output_bytes(stub_model, px=112)             # the stub's grid for one 112-px tile
    enc = BatchEncoder(*stub_model, device="cpu", compute_dtype="float32", physical_batch=8,
                       max_in_flight_bytes=2 * one_grid)
    for t in "abc":
        enc.submit(t, tiles(1, px=112))
    assert {t for t, _ in enc.drain()} >= {"a", "b"}             # the third ticket crossed the bound
    assert enc.held_bytes <= 2 * one_grid
    assert seen == [8]                                           # one padded pass, at m


def test_one_queue_per_geometry(stub_model):
    seen = record_batch_shapes(stub_model)
    enc = BatchEncoder(*stub_model, device="cpu", compute_dtype="float32", physical_batch=2)
    enc.submit("a", crops(1, px=112))
    enc.submit("b", crops(1, px=224))
    enc.flush()
    assert sorted(seen) == [(2, 3, 112, 112), (2, 3, 224, 224)]


def test_an_empty_input_completes_without_a_forward_pass(stub_model):
    seen = record_batch_sizes(stub_model)
    enc = BatchEncoder(*stub_model, device="cpu", compute_dtype="float32", physical_batch=4)
    enc.submit("empty", crops(0))
    assert [t for t, _ in enc.drain()] == ["empty"] and seen == []


def test_oom_once_halves_and_succeeds(stub_model):
    enc = BatchEncoder(*oom_once(stub_model), device="cpu", compute_dtype="float32", physical_batch=8)
    a = crops(8)
    enc.submit("a", a)
    enc.flush()
    assert enc.physical_batch == 4
    assert np.array_equal(dict(enc.drain())["a"][0], embed_alone(stub_model, a, 4))


def test_oom_twice_raises(stub_model):
    enc = BatchEncoder(*oom_always(stub_model), device="cpu", compute_dtype="float32", physical_batch=8)
    enc.submit("a", crops(8))
    with pytest.raises(Stage4OutOfMemory):
        enc.flush()
    assert enc.physical_batch == 4


def _prepared_on_the_stub(pooling, **kwargs):
    m = MeasureDinoEmbedding(pooling=pooling, device="cpu", crop_px=112, **kwargs)
    model, processor, _ = m._backbone()
    return m, model, processor


@pytest.mark.parametrize("pooling", ["crop_cls", "crop_mean", "masked_dense"])
def test_the_batch_encoder_matches_the_inline_encoder(param_stub_backbone, detected_plate, pooling):
    """The H1 promise at the encoder level, at one pass shape on both sides.

    Crop pooling: ``batch_size`` is the region count, so the inline encoder
    runs one pass of ``N`` crops, as the encoder at ``physical_batch=N`` does.
    (At ``batch_size=64`` the inline path's last pass is unpadded, and a
    multi-threaded CPU GEMM rounds a row of a 40-row pass differently from the
    same row of a 64-row one: see the next test.) ``masked_dense`` runs a tile
    a pass inline, and at 64 here.
    """
    probe, model, processor = _prepared_on_the_stub(pooling)
    n = probe.prepare_regions(detected_plate, processor=processor).n_regions
    m = probe if pooling == "masked_dense" else _prepared_on_the_stub(pooling, batch_size=n)[0]
    prepared = m.prepare_regions(detected_plate, processor=processor)
    physical = 64 if pooling == "masked_dense" else n
    inline = m.encode(prepared, m.inline_encoder(detected_plate))
    batched = m.encode(prepared, BatchEncoder(model, processor, device="cpu",
                       compute_dtype="float32", physical_batch=physical).as_region_encoder())
    assert prepared.n_regions > 64                    # more than one default batch of regions
    assert np.array_equal(inline, batched)


@pytest.mark.parametrize("pooling", ["crop_cls", "crop_mean"])
def test_only_the_pass_size_separates_inline_from_batched(param_stub_backbone, detected_plate, pooling):
    """At the default ``batch_size=64``: every row of a full inline pass is the
    encoder's bit for bit, and every row is the inline ``embed_crops``'s once
    its last pass is zero-padded to 64 as the encoder pads it (S1)."""
    from phenotypic.detect.nn._helper._dino_support import embed_crops

    m, model, processor = _prepared_on_the_stub(pooling)
    prepared = m.prepare_regions(detected_plate, processor=processor)
    n = prepared.n_regions
    full = n // 64 * 64
    assert 0 < full < n                               # a partial last pass exists
    inline = m.encode(prepared, m.inline_encoder(detected_plate))
    batched = m.encode(prepared, BatchEncoder(model, processor, device="cpu",
                       compute_dtype="float32", physical_batch=64).as_region_encoder())
    assert np.array_equal(inline[:full], batched[:full])
    (array,) = prepared.inputs.arrays
    padded = np.zeros((-(-n // 64) * 64, *array.shape[1:]), dtype=np.uint8)
    padded[:n] = array
    reference = embed_crops(model, processor, padded, pooling=pooling, device="cpu", batch_size=64)
    assert np.array_equal(reference[:n], batched)


# ---------------------------------------------------------------------------
# Beyond the plan's list: re-queue routing, the halving rule, the byte bound
# ---------------------------------------------------------------------------


def test_an_oom_requeues_rows_of_several_tickets_and_routes_them_back(stub_model):
    model, processor, calls = _raising_oom(1)
    enc = BatchEncoder(model, processor, device="cpu", compute_dtype="float32", physical_batch=4)
    a, b = crops(3, seed=1), crops(3, seed=2)
    enc.submit("a", a)
    enc.submit("b", b)  # [a0 a1 a2 b0] runs out of memory, is re-queued
    assert enc.physical_batch == 2 and enc.held_bytes > 0
    enc.flush()
    out = dict(enc.drain())
    assert calls == [4, 2, 2, 2]                    # the failed pass, then every row at m / 2
    assert np.array_equal(out["a"][0], embed_alone(stub_model, a, 2))
    assert np.array_equal(out["b"][0], embed_alone(stub_model, b, 2))
    assert enc.held_bytes == 0


def test_an_oom_inside_flush_is_retried_at_once(stub_model):
    model, processor, calls = _raising_oom(None, on_call=2)
    enc = BatchEncoder(model, processor, device="cpu", compute_dtype="float32", physical_batch=4)
    a = crops(5, seed=3)
    enc.submit("a", a)                               # [a0..a3] runs; a4 waits
    enc.flush()                                      # [a4 + 3 pads] runs out, retried at 2
    assert calls == [4, 4, 2] and enc.physical_batch == 2
    (vectors,) = dict(enc.drain())["a"]
    assert np.array_equal(vectors[:4], embed_alone(stub_model, crops(4, seed=3), 4))
    assert np.array_equal(vectors[4:], embed_alone(
        stub_model, BackboneInputs(kind="crops", arrays=(a.arrays[0][4:],), pooling="crop_cls"), 2))


def test_an_oom_at_m_1_raises_at_once(stub_model):
    enc = BatchEncoder(*oom_always(stub_model), device="cpu", compute_dtype="float32", physical_batch=1)
    with pytest.raises(Stage4OutOfMemory, match="physical batch 1"):
        enc.submit("a", crops(1))


def test_halve_halves_once_and_never_below_1(stub_model):
    enc = encoder(stub_model, 8)
    assert enc.halve() and enc.physical_batch == 4
    assert not enc.halve() and enc.physical_batch == 4
    one = encoder(stub_model, 1)
    assert not one.halve() and one.physical_batch == 1


def test_crop_mean_and_tiles_route_back_too(stub_model):
    enc = encoder(stub_model, 4)
    mean = crops(3, seed=4, pooling="crop_mean")
    grid = tiles(2, px=112, seed=5)
    enc.submit("mean", mean)
    enc.submit("grid", grid)
    enc.flush()
    out = dict(enc.drain())
    assert np.array_equal(out["mean"][0], embed_alone(stub_model, mean, 4))
    side = 112 // STUB_PATCH_SIZE
    assert [g.shape for g in out["grid"]] == [(side, side, STUB_HIDDEN_SIZE)] * 2
    assert not np.array_equal(out["grid"][0], out["grid"][1])


def test_a_crop_ticket_holds_n_d_4_bytes(stub_model):
    enc = encoder(stub_model, 8)
    enc.submit("a", crops(3))
    assert enc.held_bytes == 3 * STUB_HIDDEN_SIZE * 4
    enc.flush()
    assert enc.held_bytes == 0                       # finished; the bound counts unfinished tickets


def test_a_drained_ticket_is_dropped_and_its_id_reusable(stub_model):
    enc = encoder(stub_model, 2)
    enc.submit("a", crops(2))
    with pytest.raises(ValueError, match="already"):
        enc.submit("a", crops(1))                    # still held: not drained yet
    assert [t for t, _ in enc.drain()] == ["a"]
    assert list(enc.drain()) == []
    enc.submit("a", crops(2))                         # drained: the ticket can be used again
    assert [t for t, _ in enc.drain()] == ["a"]


def test_the_default_bound_is_the_memory_scripts(stub_model):
    assert encoder(stub_model, 2).max_in_flight_bytes == MAX_IN_FLIGHT_OUTPUT_BYTES


def test_a_model_in_another_dtype_is_refused(stub_model):
    model, processor = stub_model
    with pytest.raises(ValueError, match="bfloat16"):
        BatchEncoder(model, processor, device="cpu", compute_dtype="bfloat16", physical_batch=2)


def test_boxes_have_no_backbone(stub_model):
    boxes = BackboneInputs(kind="boxes", arrays=(np.zeros((1, 4), np.int64),), pooling="none")
    with pytest.raises(ValueError, match="boxes"):
        encoder(stub_model, 2).submit("a", boxes)


def test_a_bfloat16_model_takes_bfloat16_inputs_and_returns_float32(stub_model):
    model, processor = _build_param_stub_dino()
    model = model.to(torch.bfloat16)
    seen = _record((model, processor), lambda pixels: pixels.dtype)
    enc = BatchEncoder(model, processor, device="cpu", compute_dtype="bfloat16", physical_batch=2)
    enc.submit("a", crops(1, px=112))
    enc.flush()
    (vectors,) = dict(enc.drain())["a"]
    assert seen == [torch.bfloat16] and vectors.dtype == np.float32
