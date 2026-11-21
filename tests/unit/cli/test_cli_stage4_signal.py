"""The Stage-4 signal: one atomic Parquet per embedder key, fenced by work id, run and root.

A signal is Stage 4's per-key result for one image, written before the store
commit and consumed after the record. Its validity rule (spec §4.2, plan review
C4 and C5) is what these tests pin: a key is **valid** only when its file is
readable, fenced to this work id and run, names the key it sits under, and was
computed against the **live** store root; a file that is present but fails any
of those is **stale** and is re-embedded; an absent file is neither.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from phenotypic._cli._cli_stage4_signal import (
    SIGNAL_METADATA_KEY,
    Stage4SignalState,
    classify_stage4_signals,
    delete_stage4_signal_keys,
    delete_stage4_signals,
    read_stage4_signal,
    stage4_signal_path,
    stage4_signals_complete,
    write_stage4_signal,
)
from phenotypic.sdk_ import DIR_STAGE4_PAYLOAD, progress_dir
from phenotypic.sdk_._embedding_tables import embedding_arrow_table
from tests._tiny_run import payloads_of

_REPO = Path(__file__).resolve().parents[3]


def _classify(out: Path, keys, *, live: str, work_id: str = "w", run_id: str = "1-x"):
    return classify_stage4_signals(
        out, "ds", "img", keys, work_id=work_id, run_id=run_id, live_root_sha256=live
    )


def _rewrite_footer(path: Path, *, drop: bool = False, without: str | None = None, **changes) -> None:
    """Rewrite *path*'s footer: *drop* removes it, *without* removes one field, *changes* update it."""
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata or {})
    if drop:
        metadata.pop(SIGNAL_METADATA_KEY)
    else:
        footer = json.loads(metadata[SIGNAL_METADATA_KEY])
        footer.update(changes)
        if without is not None:
            footer.pop(without)
        metadata[SIGNAL_METADATA_KEY] = json.dumps(footer).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), path)


# ---------------------------------------------------------------------------
# The plan's tests (P3 Task 2 Step 1)
# ---------------------------------------------------------------------------


def test_a_signal_round_trips_to_the_same_store_bytes(tmp_path, image_with_objects_payload):
    payload = payloads_of(image_with_objects_payload)["toy-v1/rgb/objects"]
    path = write_stage4_signal(tmp_path, "ds", "img", payload, work_id="w", run_id="1-abc", root_sha256="r")
    back = read_stage4_signal(path, work_id="w", run_id="1-abc")
    assert back.root_sha256 == "r" and back.payload.descriptor == payload.descriptor
    assert embedding_arrow_table(back.payload).equals(embedding_arrow_table(payload))


def test_a_signal_of_another_work_id_or_run_is_ignored(tmp_path, image_with_objects_payload):
    payload = payloads_of(image_with_objects_payload)["toy-v1/rgb/objects"]
    path = write_stage4_signal(tmp_path, "ds", "img", payload, work_id="w", run_id="1-abc", root_sha256="r")
    assert read_stage4_signal(path, work_id="other", run_id="1-abc") is None
    assert read_stage4_signal(path, work_id="w", run_id="2-abc") is None


def test_the_path_is_keyed_by_model_layer_region(tmp_path):
    p = stage4_signal_path(tmp_path, "ds", "dinov2-vitb14/gray/grid-wells", "img")
    assert p.relative_to(tmp_path).parts[-6:] == (
        "stage4_payload", "ds", "dinov2-vitb14", "gray", "grid-wells", "img.parquet")


def test_signals_are_classified_per_key_against_the_live_root(tmp_path, image_with_two_models):
    a, b = payloads_of(image_with_two_models).values()
    write_stage4_signal(tmp_path, "ds", "img", a, work_id="w", run_id="1-x", root_sha256="live")
    state = classify_stage4_signals(tmp_path, "ds", "img", [a.key, b.key], work_id="w",
                                    run_id="1-x", live_root_sha256="live")
    assert state.valid == {a.key} and state.stale == frozenset()          # one key reusable
    assert not stage4_signals_complete(state, [a.key, b.key])
    write_stage4_signal(tmp_path, "ds", "img", b, work_id="w", run_id="1-x", root_sha256="old")
    state = classify_stage4_signals(tmp_path, "ds", "img", [a.key, b.key], work_id="w",
                                    run_id="1-x", live_root_sha256="live")
    assert state.valid == {a.key} and state.stale == {b.key}              # computed before a reprocess


def test_only_the_named_keys_are_deleted(tmp_path, image_with_two_models):
    a, b = payloads_of(image_with_two_models).values()
    for p in (a, b):
        write_stage4_signal(tmp_path, "ds", "img", p, work_id="w", run_id="1-x", root_sha256="r")
    assert delete_stage4_signal_keys(tmp_path, "ds", "img", [b.key]) == 1
    assert stage4_signal_path(tmp_path, "ds", a.key, "img").is_file()


def test_a_corrupt_signal_reads_as_absent(tmp_path, image_with_objects_payload, caplog):
    payload = payloads_of(image_with_objects_payload)["toy-v1/rgb/objects"]
    path = write_stage4_signal(tmp_path, "ds", "img", payload, work_id="w", run_id="1-x", root_sha256="r")
    path.write_bytes(b"corrupt")
    assert read_stage4_signal(path, work_id="w", run_id="1-x") is None
    assert str(path) in caplog.text


def test_delete_removes_every_key_of_the_stem_only(tmp_path, image_with_two_models):
    for p in payloads_of(image_with_two_models).values():
        for stem in ("img", "other"):
            write_stage4_signal(tmp_path, "ds", stem, p, work_id="w", run_id="1-x", root_sha256="r")
    assert delete_stage4_signals(tmp_path, "ds", "img") == 2
    assert len(list((tmp_path / ".phenotypic").rglob("other.parquet"))) == 2


# ---------------------------------------------------------------------------
# The validity rule, one condition at a time
# ---------------------------------------------------------------------------


def test_the_tree_segment_is_the_shared_constant(tmp_path):
    assert DIR_STAGE4_PAYLOAD == "stage4_payload"
    assert stage4_signal_path(tmp_path, "ds", "m/rgb/objects", "img") == (
        progress_dir(tmp_path) / DIR_STAGE4_PAYLOAD / "ds" / "m" / "rgb" / "objects" / "img.parquet"
    )


def test_the_footer_names_the_fence_the_root_and_the_key(tmp_path, image_with_objects_payload):
    payload = payloads_of(image_with_objects_payload)["toy-v1/rgb/objects"]
    path = write_stage4_signal(tmp_path, "ds", "img", payload, work_id="w", run_id="1-x", root_sha256="r")
    footer = json.loads(pq.read_schema(path).metadata[SIGNAL_METADATA_KEY])
    assert footer == {
        "work_id": "w", "run_id": "1-x", "root_sha256": "r",
        "key": "toy-v1/rgb/objects", "descriptor": json.loads(json.dumps(dict(payload.descriptor))),
    }
    assert pq.read_table(path).schema.field("embedding").type == pa.list_(pa.float16(), 4)


def test_an_absent_key_is_neither_valid_nor_stale(tmp_path, image_with_objects_payload):
    payload = payloads_of(image_with_objects_payload)["toy-v1/rgb/objects"]
    state = _classify(tmp_path, [payload.key], live="r")
    assert state == Stage4SignalState(valid=frozenset(), stale=frozenset())
    assert read_stage4_signal(
        stage4_signal_path(tmp_path, "ds", payload.key, "img"), work_id="w", run_id="1-x"
    ) is None


@pytest.mark.parametrize("fence", [{"work_id": "other"}, {"run_id": "2-x"}])
def test_a_signal_fenced_elsewhere_is_stale(tmp_path, image_with_objects_payload, fence):
    payload = payloads_of(image_with_objects_payload)["toy-v1/rgb/objects"]
    write_stage4_signal(tmp_path, "ds", "img", payload, work_id="w", run_id="1-x", root_sha256="r")
    state = _classify(tmp_path, [payload.key], live="r", **fence)
    assert state.valid == frozenset() and state.stale == {payload.key}


def test_a_signal_from_another_root_is_stale_and_its_root_is_reported(tmp_path, image_with_objects_payload):
    payload = payloads_of(image_with_objects_payload)["toy-v1/rgb/objects"]
    path = write_stage4_signal(tmp_path, "ds", "img", payload, work_id="w", run_id="1-x", root_sha256="old")
    assert read_stage4_signal(path, work_id="w", run_id="1-x").root_sha256 == "old"
    assert _classify(tmp_path, [payload.key], live="new").stale == {payload.key}
    assert _classify(tmp_path, [payload.key], live="old").valid == {payload.key}


def test_a_signal_without_its_footer_is_unreadable_and_stale(tmp_path, image_with_objects_payload, caplog):
    payload = payloads_of(image_with_objects_payload)["toy-v1/rgb/objects"]
    path = write_stage4_signal(tmp_path, "ds", "img", payload, work_id="w", run_id="1-x", root_sha256="r")
    _rewrite_footer(path, drop=True)
    assert read_stage4_signal(path, work_id="w", run_id="1-x") is None
    assert str(path) in caplog.text
    assert _classify(tmp_path, [payload.key], live="r").stale == {payload.key}


@pytest.mark.parametrize("field", ["work_id", "run_id", "root_sha256", "key", "descriptor"])
def test_a_footer_missing_a_field_is_unreadable_and_stale(tmp_path, image_with_objects_payload, field, caplog):
    payload = payloads_of(image_with_objects_payload)["toy-v1/rgb/objects"]
    path = write_stage4_signal(tmp_path, "ds", "img", payload, work_id="w", run_id="1-x", root_sha256="r")
    _rewrite_footer(path, without=field)
    assert read_stage4_signal(path, work_id="w", run_id="1-x") is None
    assert str(path) in caplog.text
    assert _classify(tmp_path, [payload.key], live="r").stale == {payload.key}


def test_a_footer_for_another_key_is_refused(tmp_path, image_with_two_models, caplog):
    a, b = payloads_of(image_with_two_models).values()
    path = write_stage4_signal(tmp_path, "ds", "img", a, work_id="w", run_id="1-x", root_sha256="r")
    _rewrite_footer(path, key=b.key)
    assert read_stage4_signal(path, work_id="w", run_id="1-x") is None
    assert str(path) in caplog.text
    assert _classify(tmp_path, [a.key], live="r").stale == {a.key}


def test_a_footer_whose_dim_disagrees_with_the_vectors_is_refused(tmp_path, image_with_objects_payload):
    payload = payloads_of(image_with_objects_payload)["toy-v1/rgb/objects"]
    path = write_stage4_signal(tmp_path, "ds", "img", payload, work_id="w", run_id="1-x", root_sha256="r")
    descriptor = dict(payload.descriptor) | {"dim": 8}
    _rewrite_footer(path, descriptor=descriptor)
    assert read_stage4_signal(path, work_id="w", run_id="1-x") is None


def test_a_rewrite_replaces_the_signal(tmp_path, image_with_objects_payload):
    payload = payloads_of(image_with_objects_payload)["toy-v1/rgb/objects"]
    write_stage4_signal(tmp_path, "ds", "img", payload, work_id="w", run_id="1-x", root_sha256="old")
    write_stage4_signal(tmp_path, "ds", "img", payload, work_id="w", run_id="1-x", root_sha256="new")
    assert _classify(tmp_path, [payload.key], live="new").valid == {payload.key}
    region = stage4_signal_path(tmp_path, "ds", payload.key, "img").parent
    assert sorted(p.name for p in region.iterdir()) == ["img.parquet"]


def test_complete_needs_exactly_the_keys():
    state = Stage4SignalState(valid=frozenset({"a/rgb/objects"}), stale=frozenset())
    assert stage4_signals_complete(state, ["a/rgb/objects"])
    assert not stage4_signals_complete(state, ["a/rgb/objects", "b/rgb/objects"])
    assert not stage4_signals_complete(state, [])


def test_a_stem_with_glob_characters_is_deleted_alone(tmp_path, image_with_objects_payload):
    payload = payloads_of(image_with_objects_payload)["toy-v1/rgb/objects"]
    for stem in ("plate[1]", "plate1", "plate*"):
        write_stage4_signal(tmp_path, "ds", stem, payload, work_id="w", run_id="1-x", root_sha256="r")
    assert delete_stage4_signals(tmp_path, "ds", "plate[1]") == 1
    region = stage4_signal_path(tmp_path, "ds", payload.key, "x").parent
    assert sorted(p.name for p in region.iterdir()) == ["plate*.parquet", "plate1.parquet"]


def test_deleting_nothing_returns_zero(tmp_path):
    assert delete_stage4_signals(tmp_path, "ds", "img") == 0
    assert delete_stage4_signal_keys(tmp_path, "ds", "img", ["m/rgb/objects"]) == 0


# ---------------------------------------------------------------------------
# The commit guard: every durable mutation runs under it
# ---------------------------------------------------------------------------


class _Fenced(RuntimeError):
    pass


def _fence():
    raise _Fenced("epoch fenced")


def test_a_fenced_write_publishes_nothing(tmp_path, image_with_objects_payload):
    payload = payloads_of(image_with_objects_payload)["toy-v1/rgb/objects"]
    with pytest.raises(_Fenced):
        write_stage4_signal(tmp_path, "ds", "img", payload, work_id="w", run_id="1-x",
                            root_sha256="r", commit_guard=_fence)
    region = stage4_signal_path(tmp_path, "ds", payload.key, "img").parent
    assert list(region.iterdir()) == []


@pytest.mark.parametrize("which", ["all", "keys"])
def test_a_fenced_delete_removes_nothing(tmp_path, image_with_objects_payload, which):
    payload = payloads_of(image_with_objects_payload)["toy-v1/rgb/objects"]
    path = write_stage4_signal(tmp_path, "ds", "img", payload, work_id="w", run_id="1-x", root_sha256="r")
    with pytest.raises(_Fenced):
        if which == "all":
            delete_stage4_signals(tmp_path, "ds", "img", commit_guard=_fence)
        else:
            delete_stage4_signal_keys(tmp_path, "ds", "img", [payload.key], commit_guard=_fence)
    assert path.is_file()


# ---------------------------------------------------------------------------
# Crashes: a real subprocess, killed with os._exit(137)
# ---------------------------------------------------------------------------

_CRASH_SCRIPT = textwrap.dedent(
    """
    import os, sys
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from phenotypic._cli._cli_stage4_signal import write_stage4_signal
    from phenotypic.sdk_._embedding_tables import EmbeddingPayload

    out, where = Path(sys.argv[1]), sys.argv[2]
    payload = EmbeddingPayload(
        key="toy-v1/rgb/objects",
        frame=pd.DataFrame({
            "Object_Label": np.array([1, 2, 3], dtype=np.int64),
            "embedding": [np.full(4, i, dtype=np.float16) for i in range(3)],
        }),
        descriptor={
            "model": {"family": "toy"}, "dim": 4, "dtype": "float16",
            "pooling": "box_sum", "key": "Object_Label",
            "input": {"series": "rgb", "level": 0, "layer": "rgb"},
            "producer": {"class": "ToyEmbedder", "params_sha256": "p"},
            "binds": {"objmap_path": "rgb/labels/objmap", "objmap_sha256": "o"},
        },
    )

    def _die():
        os._exit(137)

    if where == "before_replace":
        write_stage4_signal(out, "ds", "img", payload, work_id="w", run_id="1-x",
                            root_sha256="r", commit_guard=_die)
    else:
        write_stage4_signal(out, "ds", "img", payload, work_id="w", run_id="1-x",
                            root_sha256="r")
        os._exit(137)
    """
)


def _crash(out: Path, where: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _CRASH_SCRIPT, str(out), where],
        cwd=_REPO, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 137, result.stderr


def test_a_kill_after_the_write_leaves_a_valid_signal(tmp_path):
    _crash(tmp_path, "after_write")
    state = _classify(tmp_path, ["toy-v1/rgb/objects"], live="r")
    assert state.valid == {"toy-v1/rgb/objects"}
    signal = read_stage4_signal(
        stage4_signal_path(tmp_path, "ds", "toy-v1/rgb/objects", "img"), work_id="w", run_id="1-x"
    )
    assert signal.payload.frame["Object_Label"].tolist() == [1, 2, 3]


def test_a_kill_before_the_rename_leaves_no_signal(tmp_path):
    _crash(tmp_path, "before_replace")
    path = stage4_signal_path(tmp_path, "ds", "toy-v1/rgb/objects", "img")
    assert not path.exists()
    assert _classify(tmp_path, ["toy-v1/rgb/objects"], live="r") == Stage4SignalState(
        valid=frozenset(), stale=frozenset()
    )
    # The only debris is the writer's own dot-prefixed temp, which no reader opens.
    assert [p.name.startswith(".img.parquet.") for p in path.parent.iterdir()] == [True]
