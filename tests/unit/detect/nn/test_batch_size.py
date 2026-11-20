"""Stage 4's physical batch rule, ``BackboneKey`` and the cuBLAS check (#282 P2 T2).

No torch: the rule is arithmetic, and the cuBLAS check reads one variable.
The rule's constants are copied from the P0 logic-validation script, which
these tests load by path to hold the two to each other (the script never
imports ``phenotypic``; nothing forbids the reverse).
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from phenotypic.detect.nn._helper._batch_encoder import (
    ACTIVATION_FACTOR,
    CUBLAS_WORKSPACE_DEFAULT,
    DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS,
    MAX_IN_FLIGHT_OUTPUT_BYTES,
    MEMORY_FRACTION,
    RESERVE_BYTES,
    BackboneKey,
    check_cublas_workspace,
    per_sample_bytes,
    physical_batch_size,
    tokens_for,
)
from phenotypic.measure import MeasureDinoEmbedding

#: P0 T1's logic-validation scripts (``physical_batch_size.py``, ``stage4_memory.py``).
LOGIC_SCRIPTS = (
    Path(__file__).resolve().parents[4]
    / "docs/superpowers/logic_validation_scripts/2026-11-12-gpu-embedding-staging"
)


def _load_script(name: str):
    spec = spec_from_file_location(name.removesuffix(".py"), LOGIC_SCRIPTS / name)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pow2_floor_and_the_batch_size_cap():
    per = per_sample_bytes(tokens=tokens_for(224, 14, 0), hidden=768, heads=12, dtype_bytes=4)
    assert physical_batch_size(batch_size=64, gpu_total_bytes=48 << 30,
                               weight_bytes=86_000_000 * 4, per_sample=per) == 64
    m = physical_batch_size(batch_size=4096, gpu_total_bytes=48 << 30,
                            weight_bytes=86_000_000 * 4, per_sample=per)
    assert m & (m - 1) == 0 and m < 4096


def test_the_values_match_the_logic_validation_script():
    pbs = _load_script("physical_batch_size.py")
    assert (MEMORY_FRACTION, RESERVE_BYTES, ACTIVATION_FACTOR) == (
        pbs.MEMORY_FRACTION, pbs.RESERVE_BYTES, pbs.ACTIVATION_FACTOR)


def test_the_in_flight_bound_matches_the_memory_script():
    assert MAX_IN_FLIGHT_OUTPUT_BYTES == _load_script("stage4_memory.py").MAX_IN_FLIGHT_OUTPUT_BYTES


def test_every_combination_the_script_checks_gives_its_m():
    """The rule, not only its constants: the script's ``m`` for every model row,
    GPU, crop and dtype it checks, and a refusal exactly where it says 0."""
    pbs = _load_script("physical_batch_size.py")
    checked = refused = 0
    for model, (params, hidden, heads, patch, registers) in pbs.MODELS.items():
        for gpu, total in pbs.GPUS.items():
            for crop in pbs.CROPS:
                for dtype, dtype_bytes in pbs.DTYPE_BYTES.items():
                    per = per_sample_bytes(tokens=tokens_for(crop, patch, registers),
                                           hidden=hidden, heads=heads, dtype_bytes=dtype_bytes)
                    assert per == pbs.per_sample_bytes(model, crop, dtype)
                    expected = pbs.physical_batch(model, total, crop, dtype)
                    weights = int(pbs.weights_bytes(model, dtype))
                    if expected == 0:
                        with pytest.raises(ValueError, match="does not fit"):
                            physical_batch_size(batch_size=pbs.BATCH_SIZE, gpu_total_bytes=total,
                                                weight_bytes=weights, per_sample=per)
                        refused += 1
                    else:
                        assert physical_batch_size(batch_size=pbs.BATCH_SIZE, gpu_total_bytes=total,
                                                   weight_bytes=weights, per_sample=per) == expected
                    checked += 1
    assert checked == len(pbs.MODELS) * len(pbs.GPUS) * len(pbs.CROPS) * len(pbs.DTYPE_BYTES)
    assert 0 < refused < checked                      # both branches were exercised


def test_tokens_count_the_cls_and_the_registers():
    assert tokens_for(224, 14, 0) == 16 * 16 + 1
    assert tokens_for(224, 16, 4) == 14 * 14 + 1 + 4
    assert tokens_for(230, 14, 0) == 16 * 16 + 1      # the patch embedding floors


@pytest.mark.parametrize("value, ok", [(":4096:8", True), (":16:8", True), (":0:0", False), (None, False)])
def test_cublas_workspace_values(monkeypatch, value, ok):
    if value is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", value)
    if ok:
        check_cublas_workspace()
    else:
        with pytest.raises(RuntimeError, match=r"CUBLAS_WORKSPACE_CONFIG.*:4096:8.*:16:8"):
            check_cublas_workspace()


def test_the_cublas_default_is_one_of_the_accepted_values():
    assert DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS == (":4096:8", ":16:8")
    assert CUBLAS_WORKSPACE_DEFAULT == ":4096:8"


def test_weights_that_do_not_fit_are_refused():
    giant_fp32 = 1_100_000_000 * 4                                   # about 4.1 GiB of weights
    with pytest.raises(ValueError, match=r"weights\) does not fit"):  # an explicit 4 GiB card: 3.2 GiB budget
        physical_batch_size(batch_size=64, gpu_total_bytes=4 << 30,
                            weight_bytes=giant_fp32, per_sample=1)
    assert physical_batch_size(batch_size=64, gpu_total_bytes=16 << 30,
                               weight_bytes=giant_fp32, per_sample=1) == 64   # the same weights fit on 16 GiB


def test_a_sample_that_does_not_fit_is_refused():
    budget = int(MEMORY_FRACTION * (16 << 30)) - RESERVE_BYTES       # no weights
    with pytest.raises(ValueError, match="one input .* does not fit"):
        physical_batch_size(batch_size=64, gpu_total_bytes=16 << 30, weight_bytes=0,
                            per_sample=budget + 1)
    assert physical_batch_size(batch_size=64, gpu_total_bytes=16 << 30, weight_bytes=0,
                               per_sample=budget) == 1


def test_the_cap_need_not_be_a_power_of_two():
    assert physical_batch_size(batch_size=50, gpu_total_bytes=48 << 30, weight_bytes=0,
                               per_sample=1) == 50


@pytest.mark.parametrize("kwargs", [
    {"batch_size": 0}, {"gpu_total_bytes": 0}, {"weight_bytes": -1}, {"per_sample": 0}])
def test_nonsense_arguments_are_refused(kwargs):
    arguments = {"batch_size": 64, "gpu_total_bytes": 48 << 30, "weight_bytes": 0, "per_sample": 1}
    with pytest.raises(ValueError):
        physical_batch_size(**{**arguments, **kwargs})


def test_the_backbone_key_of_an_embedder():
    key = MeasureDinoEmbedding(size="large", revision="r1", compute_dtype="bfloat16").backbone_key()
    assert key == BackboneKey(version=2, size="large", revision="r1", compute_dtype="bfloat16")
    assert MeasureDinoEmbedding().backbone_key() == BackboneKey(2, "base", None, "float32")
    # Hashable, and blind to fields that do not change the weights loaded.
    assert len({MeasureDinoEmbedding(pooling="crop_mean").backbone_key(),
                MeasureDinoEmbedding(crop_px=112).backbone_key()}) == 1


def test_sort_key_orders_an_unpinned_revision_with_pinned_ones():
    keys = [BackboneKey(3, "base", None, "float32"), BackboneKey(2, "large", "b", "float32"),
            BackboneKey(2, "large", None, "float32"), BackboneKey(2, "base", "a", "bfloat16")]
    assert sorted(keys, key=BackboneKey.sort_key) == [
        BackboneKey(2, "base", "a", "bfloat16"), BackboneKey(2, "large", None, "float32"),
        BackboneKey(2, "large", "b", "float32"), BackboneKey(3, "base", None, "float32")]
