"""The durable record that stops 'best-effort' meaning 'silent'."""
from __future__ import annotations

import json
from pathlib import Path

from phenotypic.plotting._pipeline._failures import record_plot_failure
from phenotypic.sdk_._file_locking import ArtifactLockTimeout


def test_one_failure_writes_one_line(tmp_path: Path) -> None:
    record_plot_failure(
        tmp_path,
        binding_id="sym",
        plot_class="MeasureSymZones",
        lifecycle="image",
        error=TypeError("boom"),
        dataset="plate_a",
        image_stem="A01",
    )
    lines = (tmp_path / ".failures.jsonl").read_text().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["binding_id"] == "sym"
    assert entry["plot_class"] == "MeasureSymZones"
    assert entry["lifecycle"] == "image"
    assert entry["dataset"] == "plate_a"
    assert entry["image_stem"] == "A01"
    assert entry["error"] == "TypeError: boom"
    assert entry["ts"].endswith("Z")


def test_records_append_rather_than_replace(tmp_path: Path) -> None:
    for index in range(3):
        record_plot_failure(
            tmp_path,
            binding_id=f"b{index}",
            plot_class="C",
            lifecycle="measurements",
            error=ValueError(str(index)),
        )
    lines = (tmp_path / ".failures.jsonl").read_text().splitlines()
    assert [json.loads(line)["binding_id"] for line in lines] == ["b0", "b1", "b2"]


def test_aggregate_entries_omit_image_identity(tmp_path: Path) -> None:
    record_plot_failure(
        tmp_path,
        binding_id="b",
        plot_class="C",
        lifecycle="qc",
        error=ValueError("x"),
    )
    entry = json.loads((tmp_path / ".failures.jsonl").read_text())
    assert "dataset" not in entry
    assert "image_stem" not in entry


def test_recording_never_raises(tmp_path: Path) -> None:
    """A failure in the failure recorder must not escalate a soft failure."""
    unwritable = tmp_path / "nope"
    unwritable.write_text("I am a file, not a directory")

    record_plot_failure(
        unwritable,
        binding_id="b",
        plot_class="C",
        lifecycle="image",
        error=ValueError("x"),
    )  # must return normally


def test_recording_never_raises_when_the_lock_cannot_be_taken(
    tmp_path: Path, monkeypatch
) -> None:
    """The lock is a second, independent way in, inside the same handler.

    ``test_recording_never_raises`` drives a failure at the ``mkdir``, so a
    handler that only covered that first line would still pass it. This one
    leaves the directory writable and fails the *lock*, so the two tests
    cannot both be satisfied by a handler narrowed to one statement.
    """
    from phenotypic.plotting._pipeline import _failures

    def _refuse(*args, **kwargs):
        raise ArtifactLockTimeout("lock unavailable")

    monkeypatch.setattr(_failures, "exclusive_path_lock", _refuse)

    record_plot_failure(
        tmp_path,
        binding_id="b",
        plot_class="C",
        lifecycle="image",
        error=ValueError("x"),
    )  # must return normally

    assert not (tmp_path / ".failures.jsonl").exists(), (
        "the write happened before the lock was taken"
    )


def test_an_error_whose_str_raises_still_yields_a_record(
    tmp_path: Path,
) -> None:
    """Formatting the error runs user code, and it can fail.

    ``error`` is whatever a plot provider raised, so its ``__str__`` is not
    ours. Swallowing this at the top level would satisfy "never raises" and
    lose the entry -- which is the silence this module exists to remove. The
    assertion is therefore that the record *survives* with a degraded error
    field, not merely that the call returned.
    """

    class Nasty(Exception):
        def __str__(self) -> str:
            raise RuntimeError("__str__ exploded")

    record_plot_failure(
        tmp_path,
        binding_id="b",
        plot_class="C",
        lifecycle="image",
        error=Nasty(),
    )

    entry = json.loads((tmp_path / ".failures.jsonl").read_text())
    assert entry["binding_id"] == "b"
    assert entry["lifecycle"] == "image"
    assert entry["error"] == "Nasty: <unprintable: RuntimeError>"
