"""The ``compute`` block is provenance; ``compute_dtype`` is identity (#282 P2 T1).

``_shared_descriptor`` is the deliverables' one grouping function
(``_cli_embedding_deliverables.py``). It drops the per-entry ``compute``
block, as it drops ``binds`` and ``model.device``, so stores that ran on
different devices or stages share one ``descriptor_index``; and it reads a
#281 entry, which has no ``compute_dtype``, as ``"float32"``. A reduced
``compute_dtype`` moves vectors far more than a device does (H7), so it
splits the group.
"""

from __future__ import annotations

from phenotypic._cli._cli_embedding_deliverables import _shared_descriptor

_BASE = {
    "producer": {"class": "MeasureDinoEmbedding", "params_sha256": "b" * 64},
    "model": {"family": "dinov2", "revision": "a" * 40},
    "dim": 768,
    "dtype": "float16",
    "pooling": "crop_cls",
    "compute_dtype": "float32",
}


def test_compute_never_splits_a_deliverable_group():
    gpu = {**_BASE, "compute": {"device": "cuda", "stage": "stage4", "physical_batch": 64}}
    cpu = {**_BASE, "compute": {"device": "cpu", "stage": "inline", "physical_batch": 64}}
    old = {k: v for k, v in _BASE.items() if k != "compute_dtype"}  # a #281 entry
    assert _shared_descriptor(gpu) == _shared_descriptor(cpu) == _shared_descriptor(old)
    assert _shared_descriptor({**_BASE, "compute_dtype": "bfloat16"}) != _shared_descriptor(_BASE)


def test_the_shared_descriptor_carries_no_compute():
    shared = _shared_descriptor({**_BASE, "compute": {"device": "cuda"}})
    assert "compute" not in shared
    assert shared["compute_dtype"] == "float32"


def test_a_281_entry_is_not_mutated():
    old = {k: v for k, v in _BASE.items() if k != "compute_dtype"}
    _shared_descriptor(old)
    assert "compute_dtype" not in old


def test_only_a_dino_entry_reads_an_absent_compute_dtype():
    """#282 review M8: an embedder with no torch backbone never gains a ``compute_dtype``."""
    toy = {"producer": {"class": "ToyEmbedder", "params_sha256": "c" * 64},
           "model": {"family": "toy"}, "dim": 4, "dtype": "float16", "pooling": "none"}
    assert "compute_dtype" not in _shared_descriptor(toy)
    assert "compute_dtype" not in _shared_descriptor({k: v for k, v in toy.items() if k != "producer"})
