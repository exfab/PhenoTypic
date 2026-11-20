"""``MeasureDinoEmbedding`` sizes, ``compute_dtype`` and the ``compute`` block (#282 P2 T1). [torch]

DINOv2 ``giant`` (ViT-g/14) is the one new size (U11); DINOv3 H+ and 7B are
deferred, so they do not exist. ``compute_dtype`` is descriptor identity
(H7), but its ``"float32"`` default stays out of ``producer.params_sha256``,
so a #281 table does not read stale. ``compute_block`` is the one producer
of every device field (O4's condition). Torch tests skip without torch; the
rest run everywhere.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from phenotypic.measure import MeasureDinoEmbedding
from phenotypic.sdk_.typing_ import (
    ComputeDType,
    DinoSize,
    EmbeddingDinoSize,
)
from tests._tiny_run import install_stub_backbone

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="Requires torch"
)

#: ``src/phenotypic``.
SRC = Path(__file__).resolve().parents[3] / "src" / "phenotypic"

#: ``MeasureDinoEmbedding()._params_sha256()`` printed on the #281 merge
#: commit ``99bbc39cd`` (``git archive 99bbc39cd src``, imported from there).
SHA_281_DEFAULT_EMBEDDER = "a072a06fc2a26edd609f49027f8f32503bf63fb5482d1d26a1b568ff3e196f60"


@pytest.fixture(autouse=True)
def _clear_backbone_cache():
    """Tests that call :func:`install_stub_backbone` directly leave no model behind."""
    from phenotypic.measure import _measure_dino_embedding as module

    yield
    module._cached_backbone.cache_clear()


# ---------------------------------------------------------------------------
# Sizes and tokens (U11, O1, O2)
# ---------------------------------------------------------------------------


def test_the_size_and_dtype_aliases():
    assert get_args(EmbeddingDinoSize) == ("small", "base", "large", "giant")
    assert get_args(ComputeDType) == ("float32", "bfloat16", "float16")


def test_giant_with_version_3_is_refused():
    with pytest.raises(ValidationError, match="giant.*version"):
        MeasureDinoEmbedding(size="giant", version=3)
    MeasureDinoEmbedding(size="giant", version=2)  # the one valid pair


def test_the_refusal_names_the_sizes_version_3_has():
    with pytest.raises(ValidationError) as info:
        MeasureDinoEmbedding(size="giant", version=3)
    message = str(info.value)
    assert all(repr(size) in message for size in ("small", "base", "large"))


@pytest.mark.parametrize("size", ["huge_plus", "7b"])
def test_the_deferred_sizes_do_not_exist(size):
    with pytest.raises(ValidationError):  # U11
        MeasureDinoEmbedding(size=size, version=3)


def test_the_giant_token():
    assert MeasureDinoEmbedding(size="giant", version=2).model_token() == "dinov2-vitg14"


def test_the_detectors_do_not_gain_giant():
    assert get_args(DinoSize) == ("small", "base", "large")  # O1


@pytest.mark.parametrize(
    "size, version, token",
    [
        ("small", 2, "dinov2-vits14"),
        ("base", 2, "dinov2-vitb14"),
        ("large", 2, "dinov2-vitl14"),
        ("small", 3, "dinov3-vits16"),
        ("base", 3, "dinov3-vitb16"),
        ("large", 3, "dinov3-vitl16"),
    ],
)
def test_existing_model_tokens_are_unchanged(size, version, token):
    assert MeasureDinoEmbedding(size=size, version=version).model_token() == token  # O2


def test_giant_names_its_hub_repo_and_weights():
    from phenotypic.detect.nn._helper._checkpoint_manager import dino_weight_requirement
    from phenotypic.detect.nn._helper._dino_support import hf_dino_id

    assert hf_dino_id(2, "giant") == "facebook/dinov2-giant"
    assert dino_weight_requirement(2, "giant").model == "facebook/dinov2-giant"
    assert (
        MeasureDinoEmbedding(size="giant").preflight_requirements().weights[0].model
        == "facebook/dinov2-giant"
    )


# ---------------------------------------------------------------------------
# compute_dtype (H7) and the params digest
# ---------------------------------------------------------------------------


def test_default_float32_keeps_the_281_params_digest():
    assert MeasureDinoEmbedding()._params_sha256() == SHA_281_DEFAULT_EMBEDDER


def test_a_reduced_compute_dtype_is_in_the_params_digest():
    assert (
        MeasureDinoEmbedding(compute_dtype="bfloat16")._params_sha256()
        != SHA_281_DEFAULT_EMBEDDER
    )


def test_compute_dtype_is_not_tunable_and_follows_dtype():
    fields = list(MeasureDinoEmbedding.model_fields)
    assert fields.index("compute_dtype") == fields.index("dtype") + 1
    with pytest.raises(ValidationError):
        MeasureDinoEmbedding(compute_dtype="int8")


@requires_torch
def test_compute_dtype_is_descriptor_identity(param_stub_backbone, detected_plate):
    a = MeasureDinoEmbedding().payload(detected_plate).descriptor
    b = MeasureDinoEmbedding(compute_dtype="bfloat16").payload(detected_plate).descriptor
    assert (a["compute_dtype"], b["compute_dtype"]) == ("float32", "bfloat16")
    assert param_stub_backbone.load_calls[-1]["dtype"] == "bfloat16"


@requires_torch
def test_a_bfloat16_load_reaches_the_weights(param_stub_backbone, detected_plate):
    import torch

    m = MeasureDinoEmbedding(compute_dtype="bfloat16")
    m.payload(detected_plate)
    model = m._backbone()[0]
    assert next(model.parameters()).dtype == torch.bfloat16


@requires_torch
def test_two_compute_dtypes_are_two_backbones(param_stub_backbone, detected_plate):
    MeasureDinoEmbedding(device="cpu").payload(detected_plate)
    MeasureDinoEmbedding(device="cpu", compute_dtype="bfloat16").payload(detected_plate)
    MeasureDinoEmbedding(device="cpu").payload(detected_plate)  # cached
    assert [call["dtype"] for call in param_stub_backbone.load_calls] == [
        "float32",
        "bfloat16",
    ]


@requires_torch
@pytest.mark.parametrize("pooling", ["crop_cls", "crop_mean", "masked_dense"])
def test_bfloat16_vectors_are_close_to_float32(param_stub_backbone, detected_plate, pooling):
    """A bf16 forward pass runs every pooling, pools in float32, and moves the
    vectors far more than the device does (H7), but stays a close copy."""
    import numpy as np

    kwargs = {"device": "cpu", "pooling": pooling, "crop_px": 112}
    a = MeasureDinoEmbedding(**kwargs).measure(detected_plate)
    b = MeasureDinoEmbedding(compute_dtype="bfloat16", **kwargs).measure(detected_plate)
    va = np.stack(a["embedding"].to_list()).astype(np.float64)
    vb = np.stack(b["embedding"].to_list()).astype(np.float64)
    assert va.shape == vb.shape and len(va) > 0
    assert np.isfinite(vb).all()
    cosine = np.einsum("ij,ij->i", va, vb) / (
        np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1)
    )
    assert cosine.min() > 0.99
    assert not np.array_equal(va, vb)


def test_load_dino_backbone_passes_the_compute_dtype(monkeypatch):
    """``dtype=`` reaches ``AutoModel.from_pretrained`` as the string, and the
    model then moves to the device: no ``device_map``, so no ``accelerate``."""
    import sys
    import types

    from phenotypic.detect.nn._helper._dino_support import load_dino_backbone

    calls: list[tuple[str, dict]] = []
    moves: list[str] = []

    class _Model:
        def to(self, device):
            moves.append(device)
            return self

        def eval(self):
            return self

    def _recorder(kind, result):
        def from_pretrained(repo_id, **kwargs):
            calls.append((kind, kwargs))
            return result

        return types.SimpleNamespace(from_pretrained=from_pretrained)

    fake = types.ModuleType("transformers")
    fake.AutoModel = _recorder("model", _Model())
    fake.AutoImageProcessor = _recorder("processor", object())
    monkeypatch.setitem(sys.modules, "transformers", fake)

    load_dino_backbone(2, "giant", "cpu", dtype="bfloat16")
    (_, model_kwargs), (_, proc_kwargs) = calls
    assert model_kwargs["dtype"] == "bfloat16"
    assert "device_map" not in model_kwargs and "dtype" not in proc_kwargs
    assert moves == ["cpu"]


# ---------------------------------------------------------------------------
# Revisions come only from resolved_revision (#281 defect, re-review)
# ---------------------------------------------------------------------------


@requires_torch
def test_the_revision_comes_from_resolved_revision(monkeypatch, detected_plate):
    # transformers >= 5.18 sets no config._commit_hash (#281 defect; the
    # snapshot-path fallback in resolved_revision, _dino_support.py).
    install_stub_backbone(
        monkeypatch, parametric=True, commit_hash=None, snapshot_revision="d" * 40
    )
    d = MeasureDinoEmbedding(device="cpu").payload(detected_plate).descriptor
    assert d["model"]["revision"] == "d" * 40


@requires_torch
def test_the_load_revision_can_name_the_snapshot(monkeypatch, detected_plate):
    """``snapshot_from_revision=True`` lays the snapshot out at the load's
    ``revision=`` (the U5 helper P5 T2 uses)."""
    install_stub_backbone(
        monkeypatch, parametric=True, commit_hash=None, snapshot_from_revision=True
    )
    d = MeasureDinoEmbedding(device="cpu", revision="e" * 40).payload(detected_plate).descriptor
    assert d["model"]["revision"] == "e" * 40


def test_no_282_module_reads_commit_hash_directly():
    offenders = [
        p
        for p in SRC.rglob("*.py")
        if "_commit_hash" in p.read_text(encoding="utf-8") and p.name != "_dino_support.py"
    ]
    assert offenders == []  # only resolved_revision reads it


# ---------------------------------------------------------------------------
# The compute block (§4.5, O4)
# ---------------------------------------------------------------------------


@requires_torch
def test_model_and_compute_device_fields_agree(param_stub_backbone, detected_plate, monkeypatch):
    from phenotypic.detect.nn._helper import _dino_support as dino_support

    monkeypatch.setattr(dino_support, "backbone_device_name", lambda device: "NVIDIA RTX 6000 Ada")
    d = MeasureDinoEmbedding().payload(detected_plate).descriptor
    assert (d["model"]["device"], d["model"]["device_name"]) == (
        d["compute"]["device"],
        d["compute"]["device_name"],
    )  # O4's condition
    assert d["compute"]["device_name"] == "NVIDIA RTX 6000 Ada"


@requires_torch
def test_inline_entries_record_their_compute(param_stub_backbone, detected_plate):
    import torch

    compute = MeasureDinoEmbedding(device="cpu", batch_size=32).payload(detected_plate).descriptor[
        "compute"
    ]
    assert set(compute) == {
        "device", "device_name", "capability", "torch", "cuda", "physical_batch", "stage"
    }
    assert compute["stage"] == "inline" and compute["physical_batch"] == 32
    assert compute["device"] == "cpu" and compute["capability"] is None
    assert compute["device_name"] is None and compute["cuda"] is None
    assert compute["torch"] == torch.__version__


@requires_torch
def test_a_dense_inline_entry_records_one_tile_per_pass(param_stub_backbone, detected_plate):
    """``masked_dense`` runs one tile per forward pass inline, whatever ``batch_size``."""
    compute = (
        MeasureDinoEmbedding(device="cpu", pooling="masked_dense", crop_px=112, batch_size=32)
        .payload(detected_plate)
        .descriptor["compute"]
    )
    assert compute["physical_batch"] == 1


@requires_torch
def test_a_cuda_compute_block_names_the_gpu(monkeypatch):
    import torch

    from phenotypic.detect.nn._helper._dino_support import compute_block

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "NVIDIA RTX 6000 Ada")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    block = compute_block(device="cuda", stage="stage4", physical_batch=128)
    assert block == {
        "device": "cuda",
        "device_name": "NVIDIA RTX 6000 Ada",
        "capability": "8.9",
        "torch": torch.__version__,
        "cuda": "13.0",
        "physical_batch": 128,
        "stage": "stage4",
    }


@requires_torch
def test_a_given_compute_block_is_written_as_given(param_stub_backbone, detected_plate, monkeypatch):
    """A GPU stage hands ``entry_descriptor_for`` its own block; the entry's
    ``model.device`` fields are copied from it, never recomputed."""
    m = MeasureDinoEmbedding(device="cpu")
    m.payload(detected_plate)
    model = m._backbone()[0]
    block = {
        "device": "cuda", "device_name": "NVIDIA RTX PRO 6000 Blackwell", "capability": "12.0",
        "torch": "2.11.0", "cuda": "13.0", "physical_batch": 64, "stage": "stage4",
    }
    d = m.entry_descriptor_for(model, block)
    assert d["compute"] == block and d["compute"] is not block
    assert (d["model"]["device"], d["model"]["device_name"]) == ("cuda", "NVIDIA RTX PRO 6000 Blackwell")
