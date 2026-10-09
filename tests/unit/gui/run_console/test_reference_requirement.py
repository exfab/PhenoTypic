from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from phenotypic import ImagePipeline
from phenotypic._gui.run_console._app import create_app
from phenotypic._gui.run_console._callbacks import (
    _staged_gpu_capability,
    reference_metadata_requirement,
)
from phenotypic._gui.shell._sandbox import SandboxRoot
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import SubtractBlank


def _write(tmp_path, ops):
    path = tmp_path / "pipeline.json"
    path.write_text(ImagePipeline(ops=ops).to_json(), encoding="utf-8")
    return str(path)


def _callback_by_name(app: Any, name: str) -> Any:
    return next(
        callback.__wrapped__
        for spec in app.callback_map.values()
        if (callback := spec.get("callback")) is not None
        and callback.__wrapped__.__name__ == name
    )


def test_reference_pipeline_without_metadata_is_blocked(tmp_path):
    message = reference_metadata_requirement(
        _write(tmp_path, {"sb": SubtractBlank()}), None
    )
    assert message is not None and "Metadata_BlankImage" in message


def test_reference_pipeline_with_metadata_is_allowed(tmp_path):
    assert (
        reference_metadata_requirement(
            _write(tmp_path, {"sb": SubtractBlank()}), "/x/blank_map.csv"
        )
        is None
    )


def test_ordinary_or_unreadable_pipeline_is_not_blocked(tmp_path):
    assert (
        reference_metadata_requirement(
            _write(tmp_path, {"d": OtsuDetector()}), None
        )
        is None
    )
    assert (
        reference_metadata_requirement(str(tmp_path / "missing.json"), None)
        is None
    )
    assert reference_metadata_requirement(None, None) is None


def test_callback_opens_the_alert_and_run_is_disabled_until_a_table_is_chosen(
    tmp_path,
):
    app = create_app(SandboxRoot.from_path(tmp_path))
    show = _callback_by_name(app, "show_reference_metadata_requirement")
    run_disabled = _callback_by_name(app, "update_run_disabled")
    pipeline = _write(tmp_path, {"sb": SubtractBlank()})

    message, is_open = show(pipeline, {"metadata_csv": None})
    assert is_open is True and "Metadata_BlankImage" in message
    assert run_disabled(0, None, "local", False, is_open) is True

    message, is_open = show(pipeline, {"metadata_csv": "/x/blank_map.csv"})
    assert (message, is_open) == ("", False)
    assert run_disabled(0, None, "slurm", False, is_open) is False


def test_callback_stays_quiet_for_an_ordinary_pipeline(tmp_path):
    app = create_app(SandboxRoot.from_path(tmp_path))
    show = _callback_by_name(app, "show_reference_metadata_requirement")
    assert show(_write(tmp_path, {"d": OtsuDetector()}), {}) == ("", False)


def _spec(app: Any, name: str) -> tuple[str, dict[str, Any]]:
    return next(
        (key, spec)
        for key, spec in app.callback_map.items()
        if (callback := spec.get("callback")) is not None
        and callback.__wrapped__.__name__ == name
    )


def test_the_alert_and_the_run_gate_are_wired(tmp_path):
    """The callbacks above are called directly; this pins what Dash feeds them."""
    from phenotypic._gui.run_console import _ids as ids

    app = create_app(SandboxRoot.from_path(tmp_path))

    _, run_disabled = _spec(app, "update_run_disabled")
    assert {
        "id": ids.RC_REFERENCE_METADATA_REQUIRED,
        "property": "is_open",
    } in run_disabled["inputs"]

    key, show = _spec(app, "show_reference_metadata_requirement")
    assert show["inputs"] == [
        {"id": ids.RC_STORE_PIPELINE_PATH, "property": "data"},
        {"id": ids.RC_STORE_FORM_STATE, "property": "data"},
    ]
    assert f"{ids.RC_REFERENCE_METADATA_REQUIRED}.children" in key
    assert f"{ids.RC_REFERENCE_METADATA_REQUIRED}.is_open" in key


def _unknown_class_pipeline(tmp_path):
    path = tmp_path / "unknown.json"
    path.write_text(
        ImagePipeline(ops={"sb": SubtractBlank()})
        .to_json()
        .replace("SubtractBlank", "NoSuchOp"),
        encoding="utf-8",
    )
    return str(path)


def test_unknown_class_pipeline_is_not_blocked(tmp_path):
    """``from_json`` raises ``UnknownOperationClassError`` (an AttributeError)."""
    path = _unknown_class_pipeline(tmp_path)
    assert reference_metadata_requirement(path, None) is None
    assert _staged_gpu_capability(path) == (False, None)


def test_switching_to_an_unknown_class_pipeline_closes_the_alert(tmp_path):
    """Before the fix the callback raised, leaving the previous alert open."""
    app = create_app(SandboxRoot.from_path(tmp_path))
    show = _callback_by_name(app, "show_reference_metadata_requirement")

    _, is_open = show(_write(tmp_path, {"sb": SubtractBlank()}), {})
    assert is_open is True
    assert show(_unknown_class_pipeline(tmp_path), {}) == ("", False)


def test_requirement_follows_a_rewrite_that_keeps_the_mtime(tmp_path):
    """``cp -p`` / coarse mtimes: the same mtime must not serve a stale answer."""
    path = Path(_write(tmp_path, {"sb": SubtractBlank()}))
    assert reference_metadata_requirement(str(path), None) is not None

    before = path.stat()
    path.write_text(
        ImagePipeline(ops={"d": OtsuDetector()}).to_json(), encoding="utf-8"
    )
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_size != before.st_size

    assert reference_metadata_requirement(str(path), None) is None


def _output_with_snapshot(tmp_path, *, snapshot: bool) -> Path:
    """A run output; with ``snapshot``, one a previous full run left a table in."""
    from phenotypic.sdk_ import metadata_csv_deliverable_path

    output = tmp_path / "output"
    output.mkdir()
    if snapshot:
        table = metadata_csv_deliverable_path(output)
        table.parent.mkdir(parents=True)
        table.write_text("ImageName,BlankImage\nt01,t00\n", encoding="utf-8")
    return output


def test_an_output_with_a_metadata_snapshot_meets_the_requirement(tmp_path):
    """A full-mode continuation without ``--metadata`` reads the snapshot."""
    pipeline = _write(tmp_path, {"sb": SubtractBlank()})
    output = _output_with_snapshot(tmp_path, snapshot=True)
    assert reference_metadata_requirement(pipeline, None, output) is None


def test_an_output_without_a_snapshot_is_still_blocked(tmp_path):
    pipeline = _write(tmp_path, {"sb": SubtractBlank()})
    output = _output_with_snapshot(tmp_path, snapshot=False)
    message = reference_metadata_requirement(pipeline, None, output)
    assert message is not None and "Metadata_BlankImage" in message


def test_the_snapshot_is_checked_at_call_time_not_cached(tmp_path):
    """The pipeline parse is cached; whether the output has a snapshot is not."""
    from phenotypic.sdk_ import metadata_csv_deliverable_path

    pipeline = _write(tmp_path, {"sb": SubtractBlank()})
    output = _output_with_snapshot(tmp_path, snapshot=True)
    assert reference_metadata_requirement(pipeline, None, output) is None
    metadata_csv_deliverable_path(output).unlink()
    assert reference_metadata_requirement(pipeline, None, output) is not None


def test_the_alert_stays_closed_for_an_output_with_a_snapshot(tmp_path):
    app = create_app(SandboxRoot.from_path(tmp_path))
    show = _callback_by_name(app, "show_reference_metadata_requirement")
    run_disabled = _callback_by_name(app, "update_run_disabled")
    pipeline = _write(tmp_path, {"sb": SubtractBlank()})
    output = _output_with_snapshot(tmp_path, snapshot=True)

    message, is_open = show(pipeline, {"metadata_csv": None, "output_dir": str(output)})

    assert (message, is_open) == ("", False)
    assert run_disabled(0, None, "slurm", False, is_open) is False


def test_the_alert_opens_for_an_output_without_a_snapshot(tmp_path):
    """Control for the test above, through the same callback."""
    app = create_app(SandboxRoot.from_path(tmp_path))
    show = _callback_by_name(app, "show_reference_metadata_requirement")
    pipeline = _write(tmp_path, {"sb": SubtractBlank()})
    output = _output_with_snapshot(tmp_path, snapshot=False)

    message, is_open = show(pipeline, {"metadata_csv": None, "output_dir": str(output)})

    assert is_open is True and "Metadata_BlankImage" in message


def test_a_snapshot_outside_the_sandbox_does_not_count(tmp_path, tmp_path_factory):
    """The output path in the form store is client-writable; resolve it in the sandbox."""
    app = create_app(SandboxRoot.from_path(tmp_path))
    show = _callback_by_name(app, "show_reference_metadata_requirement")
    pipeline = _write(tmp_path, {"sb": SubtractBlank()})
    elsewhere = _output_with_snapshot(tmp_path_factory.mktemp("elsewhere"), snapshot=True)

    _, is_open = show(pipeline, {"metadata_csv": None, "output_dir": str(elsewhere)})

    assert is_open is True
