"""``measure_embedding_carry``: which embeddings a deferring measure core keeps (#282 U8). Torch-free.

For each embedder of the measure pipeline, the first of (1) the measure
pipeline's own run folder, (2) the run the record certifies, whose entry
under the embedder's key has the embedder's ``producer.params_sha256``;
otherwise nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from phenotypic._cli._cli_process_single import measure_embedding_carry
from phenotypic.sdk_ import ngff_, replace_image_tables
from phenotypic.sdk_._table_runs import latest_measurement_run
from tests._tiny_run import OBJECTS_ENTRY_KEY, tables_for_store

FORWARD_PARAMS = "0" * 64          # the params_sha256 store_with_crop_entry's entry records
OWN_SHA = "b" * 64


@dataclass
class _Embedder:
    key: str
    params: str

    def payload_key(self) -> str:
        return self.key

    def _params_sha256(self) -> str:
        return self.params


def _set_params(store, run_id: str, params: str) -> None:
    """Rewrite one run entry's ``producer.params_sha256`` in the root (the carry reads only that)."""
    path = store / ngff_.STORE_ROOT_JSON
    root = json.loads(path.read_text(encoding="utf-8"))
    entry = root["attributes"]["phenotypic"]["tables"]["runs"][run_id]["embeddings"][OBJECTS_ENTRY_KEY]
    entry["producer"]["params_sha256"] = params
    path.write_text(json.dumps(root), encoding="utf-8")


def _record(tmp_path, store, run_id: str):
    """A record whose ``measurements`` artifact is run *run_id*'s table (relative to *tmp_path*)."""
    path = tmp_path / "record.json"
    relative = (store / "tables" / run_id / "measurements" / "table.parquet").relative_to(tmp_path)
    path.write_text(json.dumps({"artifacts": {"measurements": {"path": str(relative)}}}), encoding="utf-8")
    return path


@pytest.fixture
def two_runs(tmp_path, store_with_crop_entry):
    """The forward run (certified) and the measure pipeline's own run, both holding the entry."""
    store = store_with_crop_entry
    forward = latest_measurement_run(store)
    own = replace_image_tables(store, tables_for_store(store), pipeline_sha256=OWN_SHA,
                               carry_embeddings=[(forward.run_id, OBJECTS_ENTRY_KEY)])
    return store, forward.run_id, own.run_id, _record(tmp_path, store, forward.run_id)


def _carry(tmp_path, store, record, *embedders):
    return measure_embedding_carry(store, record, list(embedders), output_dir=tmp_path,
                                   pipeline_sha256=OWN_SHA)


def test_the_own_run_comes_first(tmp_path, two_runs):
    store, _forward, own, record = two_runs
    assert _carry(tmp_path, store, record, _Embedder(OBJECTS_ENTRY_KEY, FORWARD_PARAMS)) == [
        (own, OBJECTS_ENTRY_KEY)]


def test_the_certified_run_when_the_own_runs_params_differ(tmp_path, two_runs):
    store, forward, own, record = two_runs
    _set_params(store, own, "e" * 64)
    assert _carry(tmp_path, store, record, _Embedder(OBJECTS_ENTRY_KEY, FORWARD_PARAMS)) == [
        (forward, OBJECTS_ENTRY_KEY)]


def test_nothing_when_neither_runs_params_match(tmp_path, two_runs):
    store, _forward, _own, record = two_runs
    assert _carry(tmp_path, store, record, _Embedder(OBJECTS_ENTRY_KEY, "f" * 64)) == []


def test_nothing_for_a_key_no_run_declares(tmp_path, two_runs):
    store, _forward, _own, record = two_runs
    assert _carry(tmp_path, store, record, _Embedder("dinov2-vitb14/rgb/grid", FORWARD_PARAMS)) == []


def test_the_certified_run_alone_without_an_own_folder(tmp_path, store_with_crop_entry):
    store = store_with_crop_entry
    forward = latest_measurement_run(store).run_id
    record = _record(tmp_path, store, forward)
    assert _carry(tmp_path, store, record, _Embedder(OBJECTS_ENTRY_KEY, FORWARD_PARAMS)) == [
        (forward, OBJECTS_ENTRY_KEY)]


def test_a_run_the_record_does_not_certify_is_not_a_source(tmp_path, store_with_crop_entry):
    """Rule 2 is the certified run, never merely another run of the store."""
    store = store_with_crop_entry
    assert _carry(tmp_path, store, tmp_path / "no-record.json",
                  _Embedder(OBJECTS_ENTRY_KEY, FORWARD_PARAMS)) == []
