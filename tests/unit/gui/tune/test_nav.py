from phenotypic.gui.tune._nav import (
    DESTINATIONS,
    active_destination,
    destination_button_class,
    destination_button_disabled,
    destination_view_class,
)


def test_destinations_are_setup_run_monitor_in_order():
    assert DESTINATIONS == ("setup", "run", "monitor")


def test_active_destination_maps_trigger_to_name():
    assert active_destination(
        "tune-dest-run", pipeline_path="pipeline.json.pht-pipe"
    ) == "run"
    assert active_destination("tune-dest-monitor") == "monitor"


def test_active_destination_keeps_run_inert_without_pipeline():
    assert active_destination("tune-dest-run", pipeline_path=None) == "setup"
    assert active_destination("tune-dest-run", pipeline_path="pipeline.json.pht-pipe") == "run"


def test_active_destination_defaults_to_setup_on_none():
    assert active_destination(None) == "setup"


def test_classes_mark_active_and_hide_inactive():
    assert "tune-dest-active" in destination_button_class("setup", "setup")
    assert "tune-dest-active" not in destination_button_class("run", "setup")
    assert "tune-view-hidden" in destination_view_class("run", "setup")
    assert "tune-view-hidden" not in destination_view_class("setup", "setup")


def test_run_destination_button_is_disabled_until_pipeline_exists():
    assert destination_button_disabled("run", pipeline_path=None) is True
    assert destination_button_disabled("run", pipeline_path="pipeline.json.pht-pipe") is False
    assert destination_button_disabled("setup", pipeline_path=None) is False
