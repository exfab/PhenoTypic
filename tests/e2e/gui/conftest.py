"""Pytest fixtures + gating for browser-driven GUI E2E tests.

These tests require Playwright + Chromium. They are skipped unless the
``PLAYWRIGHT`` environment variable is set to ``1`` (matches the CI gate in
``.github/workflows/gui-e2e.yml``).

Phase 0 only ships the gating skip. Server, sandbox, and browser fixtures
land in Phase 3 (shell smoke) and Phase 5 (composed mounts). See
``GUI_SPEC_V1.md`` section 8 (Testing).
"""
from __future__ import annotations

import os

import pytest

if os.environ.get("PLAYWRIGHT") != "1":
    pytest.skip(
        "Set PLAYWRIGHT=1 to run browser E2E tests "
        "(CI sets this automatically when gui-e2e workflow triggers).",
        allow_module_level=True,
    )

# TODO(Phase 3): server-on-ephemeral-port fixture.
# TODO(Phase 3): fake_sandbox fixture (synthetic plate fixture under tmp_path).
# TODO(Phase 3): browser_page fixture (pytest-playwright provides ``page``).
