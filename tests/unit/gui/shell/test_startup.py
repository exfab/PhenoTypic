"""Unit tests for the launcher startup reporter (``gui/shell/_startup.py``)."""
from __future__ import annotations

import io
import logging

import pytest

from phenotypic.gui.shell._startup import StartupReporter, should_use_rich


class _FakeTTY(io.StringIO):
    """StringIO that claims to be a TTY (or not) for ``should_use_rich``."""

    def __init__(self, *, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:  # noqa: D401 - trivial
        return self._tty


class TestShouldUseRich:
    def test_tty_non_debug_uses_rich(self) -> None:
        assert should_use_rich(debug=False, stream=_FakeTTY(tty=True)) is True

    def test_non_tty_falls_back_to_plain(self) -> None:
        assert should_use_rich(debug=False, stream=_FakeTTY(tty=False)) is False

    def test_debug_disables_rich_even_on_tty(self) -> None:
        assert should_use_rich(debug=True, stream=_FakeTTY(tty=True)) is False

    def test_no_progress_env_disables_rich(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PHENOTYPIC_GUI_NO_PROGRESS", "1")
        assert should_use_rich(debug=False, stream=_FakeTTY(tty=True)) is False


class TestStartupReporterPlain:
    """Plain (non-rich) mode emits one log line per stage, in order."""

    def test_stage_sequence_logs_in_order(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        reporter = StartupReporter(
            total_steps=3, use_rich=False, import_elapsed=1.23
        )
        with caplog.at_level(logging.INFO, logger="phenotypic.gui.startup"):
            with reporter:
                reporter.record_done(
                    "Core library loaded", reporter.import_elapsed
                )
                with reporter.stage("Resolving sandbox root"):
                    pass
                with reporter.stage("Composing GUI hub") as r:
                    r.detail("builder")  # sub-step → debug, no INFO line

        messages = [rec.getMessage() for rec in caplog.records]
        text = "\n".join(messages)
        assert "Core library loaded" in text
        assert "1.23s" in text  # measured import elapsed surfaced verbatim
        assert "Resolving sandbox root" in messages[1]
        # Completion line carries the ✓ marker.
        assert any("✓" in m and "Resolving sandbox root" in m for m in messages)
        assert any("Composing GUI hub" in m for m in messages)

    def test_stage_failure_logs_error_and_reraises(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        reporter = StartupReporter(total_steps=1, use_rich=False)
        with caplog.at_level(logging.ERROR, logger="phenotypic.gui.startup"):
            with pytest.raises(ValueError):
                with reporter:
                    with reporter.stage("Resolving sandbox root"):
                        raise ValueError("bad root")
        assert any("✗" in r.getMessage() for r in caplog.records)

    def test_record_done_without_elapsed_omits_timing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        reporter = StartupReporter(total_steps=1, use_rich=False)
        with caplog.at_level(logging.INFO, logger="phenotypic.gui.startup"):
            with reporter:
                reporter.record_done("Starting server", None)
        msg = caplog.records[0].getMessage()
        assert "Starting server" in msg and "s)" not in msg


class TestStartupReporterRich:
    """Rich mode quiets INFO logging while the live bar owns the terminal."""

    def test_rich_mode_restores_log_level(self) -> None:
        root = logging.getLogger()
        original = root.level
        root.setLevel(logging.INFO)
        try:
            reporter = StartupReporter(total_steps=2, use_rich=True)
            with reporter:
                # Inside the live bar, INFO is suppressed to avoid corrupting it.
                assert root.level == logging.WARNING
                reporter.record_done("Core library loaded", 1.0)
                with reporter.stage("Composing GUI hub") as r:
                    r.detail("viewer")
            # Restored on exit.
            assert root.level == logging.INFO
        finally:
            root.setLevel(original)
