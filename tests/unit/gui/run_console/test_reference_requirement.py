from __future__ import annotations

from typing import Any

from phenotypic import ImagePipeline
from phenotypic._gui.run_console._app import create_app
from phenotypic._gui.run_console._callbacks import (
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
