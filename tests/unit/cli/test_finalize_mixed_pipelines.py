"""#281 P3 review OQ4 (C16r): finalize advises when certified folders mix pipelines.

Decided as warn, not refuse. After ``--mode measure`` with B, a continuation
with A that adds images certifies B's folder for the old images and A's for
the new one (spec D6a), so the master mixes the two pipelines' columns. That
is allowed; finalize names the pipelines, and ``metrics.json`` reports a
``null`` ``pipeline_sha256``.
"""

from __future__ import annotations

import json
import logging
import shutil

from phenotypic._cli._cli_failure_tracker import file_sha256
from phenotypic._cli._cli_finalize_run import _warn_on_mixed_pipelines
from tests._tiny_run import run_cli

_FINALIZE_LOGGER = "phenotypic._cli._cli_finalize_run"
SHA_A = "a" * 64
SHA_B = "b" * 64


def _mixed_messages(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == _FINALIZE_LOGGER and "pipelines" in record.getMessage()
    ]


def test_mixed_pipelines_are_named(caplog):
    with caplog.at_level(logging.WARNING, logger=_FINALIZE_LOGGER):
        _warn_on_mixed_pipelines([SHA_A, SHA_B, SHA_B, None])
    (message,) = _mixed_messages(caplog)
    assert "2 pipelines" in message
    assert f"{SHA_A[:12]} (1 table(s))" in message
    assert f"{SHA_B[:12]} (2 table(s))" in message


def test_one_pipeline_or_unknown_digests_say_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger=_FINALIZE_LOGGER):
        _warn_on_mixed_pipelines([SHA_A, SHA_A])
        _warn_on_mixed_pipelines([SHA_A, None])
        _warn_on_mixed_pipelines([])
    assert _mixed_messages(caplog) == []


def test_a_continuation_after_measure_b_warns_and_still_finishes(
    finished_tiny_run, tiny_inputs, pipeline_file_b, caplog
):
    from phenotypic.sdk_ import metrics_json_path

    out = finished_tiny_run
    result = run_cli(None, out, mode="measure", pipeline=pipeline_file_b)
    assert result.exit_code == 0, result.output
    shutil.copy2(tiny_inputs / "ds" / "plate_1.tif", tiny_inputs / "ds" / "plate_3.tif")

    with caplog.at_level(logging.WARNING, logger=_FINALIZE_LOGGER):
        result = run_cli(tiny_inputs, out)
    assert result.exit_code == 0, result.output  # warn, never refuse
    (message,) = _mixed_messages(caplog)
    sha_a = file_sha256(out.parent / "tiny_run.pht-pipe.json")
    sha_b = file_sha256(pipeline_file_b)
    assert f"{sha_a[:12]} (1 table(s))" in message
    assert f"{sha_b[:12]} (2 table(s))" in message
    metrics = json.loads(metrics_json_path(out).read_text(encoding="utf-8"))
    assert metrics["pipeline_sha256"] is None
