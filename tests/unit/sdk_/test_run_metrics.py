"""``metrics.json``'s builders (#281 spec C4): shape, canonical bytes, the shared sha."""

from __future__ import annotations

import json

import pytest

from phenotypic.sdk_._run_metrics import (
    RUN_METRICS_SCHEMA_VERSION,
    build_run_metrics,
    shared_pipeline_sha256,
    write_canonical_json,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def _metrics(**overrides):
    values = dict(
        phenotypic_version="0.21.0",
        pipeline_sha256=SHA_A,
        accepted=3,
        succeeded=2,
        failed=1,
        measured_objects=180,
        metadata_phantoms=4,
    )
    values.update(overrides)
    return build_run_metrics(**values)


def test_metrics_has_no_wall_clock_and_is_canonical(tmp_path):
    m = _metrics()
    write_canonical_json(tmp_path / "m.json", m)
    text = (tmp_path / "m.json").read_text(encoding="utf-8")
    assert text.endswith("}\n") and "utc" not in text.lower()
    assert json.loads(text)["images"] == {"accepted": 3, "succeeded": 2, "failed": 1}
    assert "embeddings" not in m


def test_metrics_carries_every_spec_key():
    assert _metrics() == {
        "schema_version": RUN_METRICS_SCHEMA_VERSION,
        "phenotypic_version": "0.21.0",
        "pipeline_sha256": SHA_A,
        "images": {"accepted": 3, "succeeded": 2, "failed": 1},
        "objects": {"measured": 180},
        "metadata": {"phantoms": 4},
    }


def test_canonical_json_is_sorted_and_indented(tmp_path):
    path = tmp_path / "m.json"
    write_canonical_json(path, {"b": 1, "a": {"d": 2, "c": "é"}})
    assert path.read_bytes() == (
        '{\n  "a": {\n    "c": "é",\n    "d": 2\n  },\n  "b": 1\n}\n'
    ).encode("utf-8")


def test_the_same_metrics_give_the_same_bytes(tmp_path):
    write_canonical_json(tmp_path / "one.json", _metrics())
    write_canonical_json(tmp_path / "two.json", _metrics())
    assert (tmp_path / "one.json").read_bytes() == (tmp_path / "two.json").read_bytes()


def test_the_embeddings_hook_is_present_only_when_given():
    m = _metrics(embeddings={"dinov2-vitb14/crops": {"rows": 7, "stale_skipped": 1}})
    assert m["embeddings"] == {"dinov2-vitb14/crops": {"rows": 7, "stale_skipped": 1}}


def test_an_unknown_pipeline_sha_is_written_as_null(tmp_path):
    write_canonical_json(tmp_path / "m.json", _metrics(pipeline_sha256=None))
    assert json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))["pipeline_sha256"] is None


@pytest.mark.parametrize(
    ("digests", "expected"),
    [
        ([SHA_A, SHA_A, SHA_A], SHA_A),
        ([SHA_A, SHA_B], None),
        ([SHA_A, None], None),
        ([None, None], None),
        ([], None),
    ],
)
def test_the_pipeline_sha_is_reported_only_when_every_table_shares_it(digests, expected):
    assert shared_pipeline_sha256(digests) == expected
