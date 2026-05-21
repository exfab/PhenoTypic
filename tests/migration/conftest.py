"""Pytest configuration for the pydantic-migration equivalence suite.

This suite is deliberately decoupled from ``tests/unit/conftest.py`` --
its inputs come from the frozen ``_inputs/`` artifacts, not the session
fixtures, so it keeps working once the operation tree is migrated.
"""
