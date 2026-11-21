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
    MEMORY_FRACTION,
    RESERVE_BYTES,
    BatchEncoder,
    Stage4OutOfMemory,
    per_sample_bytes,
    physical_batch_for,
    physical_batch_size,
    tokens_for,
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


def outputs_of(enc):
    """``{ticket: outputs}`` of every finished ticket."""
    return {done.ticket: done.outputs for done in enc.drain()}


def only_m(enc):
    """The one ``m`` of an encoder whose rows all sat in one queue."""
    (m,) = set(enc.physical_batches.values())
    return m


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
    out = outputs_of(enc)
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
        vectors.append(outputs_of(enc)["p"][0])
    return all(np.array_equal(v, vectors[0]) for v in vectors)


def test_a_row_does_not_depend_on_its_neighbours_on_cpu(stub_model):
    assert rows_are_isolated(stub_model)


def test_the_isolation_check_catches_a_leaky_model(leaky_stub_model):
    assert not rows_are_isolated(leaky_stub_model)              # the check above can fail (plan review I13)


def test_a_finished_image_is_drained_before_a_partial_batch(stub_model):
    enc = BatchEncoder(*stub_model, device="cpu", compute_dtype="float32", physical_batch=4)
    enc.submit("full", crops(4))
    enc.submit("partial", crops(1))
    assert [d.ticket for d in enc.drain()] == ["full"]              # not held behind "partial"


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
    enc.submit("a", tiles(1, px=112))
    enc.submit("b", tiles(1, px=112))
    assert list(enc.drain()) == [] and seen == []                # at the bound is not over it
    assert enc.held_bytes == 2 * one_grid
    enc.submit("c", tiles(1, px=112))                            # the third ticket crosses it
    assert {d.ticket for d in enc.drain()} == {"a", "b", "c"}
    assert enc.held_bytes == 0
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
    assert [d.ticket for d in enc.drain()] == ["empty"] and seen == []


def test_oom_once_halves_and_succeeds(stub_model):
    enc = BatchEncoder(*oom_once(stub_model), device="cpu", compute_dtype="float32", physical_batch=8)
    a = crops(8)
    enc.submit("a", a)
    enc.flush()
    assert only_m(enc) == 4
    assert np.array_equal(outputs_of(enc)["a"][0], embed_alone(stub_model, a, 4))


def test_oom_twice_raises(stub_model):
    enc = BatchEncoder(*oom_always(stub_model), device="cpu", compute_dtype="float32", physical_batch=8)
    enc.submit("a", crops(8))
    with pytest.raises(Stage4OutOfMemory):
        enc.flush()
    assert only_m(enc) == 4


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
    a pass inline, and so at ``physical_batch=1`` here: the same pass shape,
    not CPU GEMM luck (review M3).
    """
    probe, model, processor = _prepared_on_the_stub(pooling)
    n = probe.prepare_regions(detected_plate, processor=processor).n_regions
    m = probe if pooling == "masked_dense" else _prepared_on_the_stub(pooling, batch_size=n)[0]
    prepared = m.prepare_regions(detected_plate, processor=processor)
    physical = 1 if pooling == "masked_dense" else n  # inline runs one tile a pass (review M3)
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
    assert only_m(enc) == 2 and enc.held_bytes > 0
    enc.flush()
    out = outputs_of(enc)
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
    assert calls == [4, 4, 2] and only_m(enc) == 2
    (vectors,) = outputs_of(enc)["a"]
    assert np.array_equal(vectors[:4], embed_alone(stub_model, crops(4, seed=3), 4))
    assert np.array_equal(vectors[4:], embed_alone(
        stub_model, BackboneInputs(kind="crops", arrays=(a.arrays[0][4:],), pooling="crop_cls"), 2))


def test_an_oom_at_m_1_raises_at_once(stub_model):
    enc = BatchEncoder(*oom_always(stub_model), device="cpu", compute_dtype="float32", physical_batch=1)
    with pytest.raises(Stage4OutOfMemory, match="physical batch 1"):
        enc.submit("a", crops(1))


def test_crop_mean_and_tiles_route_back_too(stub_model):
    enc = encoder(stub_model, 4)
    mean = crops(3, seed=4, pooling="crop_mean")
    grid = tiles(2, px=112, seed=5)
    enc.submit("mean", mean)
    enc.submit("grid", grid)
    enc.flush()
    out = outputs_of(enc)
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
    assert [d.ticket for d in enc.drain()] == ["a"]
    assert list(enc.drain()) == []
    enc.submit("a", crops(2))                         # drained: the ticket can be used again
    assert [d.ticket for d in enc.drain()] == ["a"]


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
    (vectors,) = outputs_of(enc)["a"]
    assert seen == [torch.bfloat16] and vectors.dtype == np.float32


# ---------------------------------------------------------------------------
# #282 P1/P2 review: per-ticket pass sizes (I1), m per queue (I2), a poisoned encoder (I6)
# ---------------------------------------------------------------------------


def test_a_halving_mid_ticket_records_every_pass_size():
    """I1: a ticket whose rows ran at 4 and then, after an OOM, at 2 reports both."""
    from phenotypic.detect.nn._helper._dino_support import compute_block

    model, processor, calls = _raising_oom(None, on_call=2)
    enc = BatchEncoder(model, processor, device="cpu", compute_dtype="float32", physical_batch=4)
    enc.submit("a", crops(6, seed=1))                 # [a0..a3] runs at 4; a4, a5 wait
    enc.flush()                                       # [a4 a5 + 2 pads] runs out; retried at 2
    enc.submit("b", crops(2, seed=2))                 # after the halving: at 2 only
    enc.flush()
    drained = {d.ticket: d.physical_batches for d in enc.drain()}
    assert calls == [4, 4, 2, 2]
    assert drained == {"a": (4, 2), "b": (2,)}
    block = compute_block(device="cpu", stage="stage4", physical_batch=drained["a"])
    assert block["physical_batch"] is None and block["physical_batches"] == [4, 2]
    assert compute_block(device="cpu", stage="stage4", physical_batch=drained["b"])["physical_batch"] == 2


def test_a_zero_row_ticket_ran_no_pass(stub_model):
    enc = encoder(stub_model, 4)
    enc.submit("empty", crops(0))
    assert [d.physical_batches for d in enc.drain()] == [()]


def _stub_heads(model, heads=1):
    """The stub's config has no attention heads; the physical-batch rule reads them."""
    model.config.num_attention_heads = heads
    return model


def test_embedders_sharing_a_backbone_get_an_m_per_crop_px(param_stub_backbone, detected_plate):
    """I2: one model load, two ``crop_px``; each queue runs at its own geometry's ``m``."""
    small = MeasureDinoEmbedding(device="cpu", crop_px=112)
    big = MeasureDinoEmbedding(device="cpu", crop_px=224)
    assert small.backbone_key() == big.backbone_key()          # one load serves both (H10)
    model, processor, _ = small._backbone()
    _stub_heads(model)
    weights = sum(p.numel() * p.element_size() for p in model.parameters())

    def per(px):
        return per_sample_bytes(tokens=tokens_for(px, STUB_PATCH_SIZE, 0), hidden=STUB_HIDDEN_SIZE,
                                heads=1, dtype_bytes=4)

    # A GPU on which two and a half 224-px inputs fit beside the weights.
    total = int((RESERVE_BYTES + weights + 2.5 * per(224)) / MEMORY_FRACTION) + 1

    def rule(h, w):
        return physical_batch_for(model, crop_px=(h, w), batch_size=64,
                                  compute_dtype="float32", gpu_total_bytes=total)

    m_small, m_big = rule(112, 112), rule(224, 224)
    assert m_big == 2 and m_small > m_big                      # the rule depends on crop_px
    assert m_small == physical_batch_size(batch_size=64, gpu_total_bytes=total,
                                          weight_bytes=weights, per_sample=per(112))
    seen = record_batch_shapes((model, processor))
    enc = BatchEncoder(model, processor, device="cpu", compute_dtype="float32", physical_batch=rule)
    enc.submit("small", small.prepare_regions(detected_plate).inputs)
    enc.submit("big", big.prepare_regions(detected_plate).inputs)
    enc.flush()
    assert {s[0] for s in seen if s[2] == 112} == {m_small}
    assert {s[0] for s in seen if s[2] == 224} == {m_big}
    assert {d.ticket: d.physical_batches for d in enc.drain()} == {"small": (m_small,), "big": (m_big,)}


def test_an_oom_halves_only_its_own_queue(stub_model):
    """I2: running out at 224 px halves the 224-px queue; the 112-px queue keeps its m."""
    model, processor = stub_model
    calls = []

    def hook(module, args, kwargs):
        shape = tuple(kwargs["pixel_values"].shape)
        calls.append(shape)
        if shape[2] == 224 and sum(c[2] == 224 for c in calls) == 1:
            raise torch.OutOfMemoryError("CUDA out of memory. (test)")

    model.register_forward_pre_hook(hook, with_kwargs=True)
    enc = encoder(stub_model, 4)
    enc.submit("small", crops(4, px=112, seed=1))
    enc.submit("big", crops(4, px=224, seed=2))        # runs out at 4; halved to 2, re-queued
    enc.submit("small2", crops(4, px=112, seed=3))
    enc.flush()
    assert enc.physical_batches == {("crops", 112, 112, "crop_cls"): 4, ("crops", 224, 224, "crop_cls"): 2}
    assert [c[0] for c in calls if c[2] == 112] == [4, 4]
    drained = {d.ticket: d.physical_batches for d in enc.drain()}
    assert drained == {"small": (4,), "big": (2,), "small2": (4,)}


def test_a_rule_that_gives_no_positive_m_is_refused(stub_model):
    enc = encoder(stub_model, lambda h, w: 0)
    with pytest.raises(ValueError, match="positive int"):
        enc.submit("a", crops(1, px=112))
    assert enc.held_bytes == 0 and list(enc.drain()) == []


def test_a_non_oom_forward_error_requeues_its_rows_and_poisons_the_encoder(stub_model):
    """I6: a pass holding rows of two tickets fails; no row is lost, and the encoder refuses use."""
    model, processor = stub_model
    calls = []

    def hook(module, args, kwargs):
        calls.append(int(kwargs["pixel_values"].shape[0]))
        if len(calls) == 2:
            raise ValueError("an illegal memory access (test)")

    model.register_forward_pre_hook(hook, with_kwargs=True)
    enc = encoder(stub_model, 4)
    enc.submit("a", crops(4, seed=1))                  # pass 1 runs: "a" is done
    enc.submit("b", crops(3, seed=2))
    held = enc.held_bytes
    with pytest.raises(ValueError, match="illegal memory access"):
        enc.submit("c", crops(3, seed=3))              # pass 2, [b0 b1 b2 c0], fails
    assert enc.held_bytes == held + 3 * STUB_HIDDEN_SIZE * 4
    (queue,) = enc._queues.values()
    assert [(t, i) for t, i, _ in queue] == [("b", 0), ("b", 1), ("b", 2), ("c", 0), ("c", 1), ("c", 2)]
    for use in (lambda: enc.submit("d", crops(1)), enc.flush, enc.drain):
        with pytest.raises(RuntimeError, match="unusable") as info:
            use()
        assert isinstance(info.value.__cause__, ValueError)
    assert calls == [4, 4]                             # nothing ran after the failure



# ---------------------------------------------------------------------------
# #282 review: a bottomed-out OOM poisons (lead ruling), M7, M14
# ---------------------------------------------------------------------------


def test_an_oom_that_cannot_halve_requeues_and_poisons_the_encoder():
    model, processor, calls = _raising_oom(None)
    enc = BatchEncoder(model, processor, device="cpu", compute_dtype="float32", physical_batch=2)
    enc.submit("a", crops(1, seed=1))
    enc.submit("b", crops(1, seed=2))                  # [a0 b0] runs out: halved to 1, re-queued
    held = enc.held_bytes
    with pytest.raises(Stage4OutOfMemory, match="after halving once"):
        enc.flush()                                    # [a0] runs out at 1: cannot halve
    assert enc.held_bytes == held == 2 * STUB_HIDDEN_SIZE * 4
    (queue,) = enc._queues.values()
    assert [(t, i) for t, i, _ in queue] == [("a", 0), ("b", 0)]
    for use in (lambda: enc.submit("c", crops(1)), enc.flush, enc.drain):
        with pytest.raises(RuntimeError, match="unusable") as info:
            use()
        assert isinstance(info.value.__cause__, Stage4OutOfMemory)
    assert calls == [2, 1]


def test_only_a_device_oom_halves():
    from phenotypic.detect.nn._helper._batch_encoder import _is_device_oom

    assert _is_device_oom(torch.OutOfMemoryError("CUDA out of memory."))
    assert not _is_device_oom(MemoryError())           # the host's: ends the shard (M7)
    assert not _is_device_oom(RuntimeError("CUDA error: an illegal memory access"))


def test_an_unknown_compute_dtype_is_refused(stub_model):
    with pytest.raises(ValueError, match="compute_dtype must be one of"):
        BatchEncoder(*stub_model, device="cpu", compute_dtype="int8", physical_batch=2)


@pytest.mark.parametrize("inputs, match", [
    (BackboneInputs(kind="crops", arrays=(np.zeros((1, 112, 112, 3), np.float32),), pooling="crop_cls"), "uint8"),
    (BackboneInputs(kind="crops", arrays=(np.zeros((1, 112, 112, 4), np.uint8),), pooling="crop_cls"), "uint8"),
    (BackboneInputs(kind="crops", arrays=(np.zeros((1, 112, 112, 3), np.uint8),), pooling="patches"), "crop_cls"),
    (BackboneInputs(kind="tiles", arrays=(np.zeros((4, 112, 112), np.float32),), pooling="patches"), r"\(3, h, w\)"),
    (BackboneInputs(kind="tiles", arrays=(np.zeros((3, 112, 112), np.float64),), pooling="patches"), "float32"),
    (BackboneInputs(kind="tiles", arrays=(np.zeros((3, 112, 112), np.float32),), pooling="crop_cls"), "patches"),
])
def test_inputs_that_are_not_a_backbones_are_refused(stub_model, inputs, match):
    """M14: ``_admit``'s refusals; a refused submission holds nothing."""
    enc = encoder(stub_model, 2)
    with pytest.raises(ValueError, match=match):
        enc.submit("a", inputs)
    assert enc.held_bytes == 0 and list(enc.drain()) == []


def test_a_batched_tile_is_squeezed(stub_model):
    """M14: a ``(1, 3, h, w)`` tile, as an HF processor returns it, runs as ``(3, h, w)``."""
    tile = np.random.default_rng(6).normal(size=(3, 112, 112)).astype(np.float32)
    outputs = []
    for array in (tile, tile[None]):
        enc = encoder(stub_model, 2)
        enc.submit("t", BackboneInputs(kind="tiles", arrays=(array,), pooling="patches"))
        enc.flush()
        outputs.append(outputs_of(enc)["t"][0])
    assert np.array_equal(outputs[0], outputs[1])


def test_the_sync_adaptor_refuses_an_encoder_holding_another_ticket(stub_model):
    enc = encoder(stub_model, 4)
    enc.submit("held", crops(1))
    with pytest.raises(RuntimeError, match="holds no other ticket"):
        enc.as_region_encoder().run(crops(1, seed=1))


def test_bfloat16_dense_batched_matches_inline_at_one_pass_shape(param_stub_backbone, detected_plate):
    """M14: bf16 ``masked_dense``, inline against the encoder at ``physical_batch=1``: the
    tile is cast float32 -> bfloat16 on both sides, so the vectors are identical."""
    m = MeasureDinoEmbedding(pooling="masked_dense", crop_px=112, device="cpu", compute_dtype="bfloat16")
    model, processor, _ = m._backbone()
    prepared = m.prepare_regions(detected_plate, processor=processor)
    inline = m.encode(prepared, m.inline_encoder(detected_plate))
    batched = m.encode(prepared, BatchEncoder(model, processor, device="cpu", compute_dtype="bfloat16",
                                              physical_batch=1).as_region_encoder())
    assert prepared.n_regions > 0 and np.array_equal(inline, batched)
