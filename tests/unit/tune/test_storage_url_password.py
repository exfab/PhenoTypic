"""``_resolve_storage_url`` rejects a password-bearing storage URL (B2).

The single chokepoint that resolves the Optuna storage URL also refuses an
inline password: that secret would otherwise be persisted in plaintext into the
GUI-discovery ``run.json`` marker AND the generated SLURM worker script, both
world-readable on a shared cluster filesystem. A password-less Postgres URL
(libpq resolves the secret from ``~/.pgpass`` / ``$PGPASSWORD`` / ``PGSERVICE``)
and the local SQLite fallback must still resolve cleanly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic.tune._strategies._config import PHENOTYPIC_TUNE_STORAGE_URL_ENV
from phenotypic.tune._tune_cli._run import _resolve_storage_url


def test_rejects_explicit_password_bearing_url(tmp_path):
    with pytest.raises(ValueError, match="inline password"):
        _resolve_storage_url(
            "postgresql+psycopg://user:s3cret@db.example.org:5432/tune",
            tmp_path,
        )


def test_rejects_password_bearing_url_from_env(tmp_path, monkeypatch):
    # The env-var fallback is the same chokepoint — a secret there is rejected too.
    monkeypatch.setenv(
        PHENOTYPIC_TUNE_STORAGE_URL_ENV,
        "postgresql+psycopg://user:s3cret@db.example.org:5432/tune",
    )
    with pytest.raises(ValueError, match="inline password"):
        _resolve_storage_url(None, tmp_path)


def test_password_less_postgres_url_passes(tmp_path):
    url = "postgresql+psycopg://user@db.example.org:5432/tune"
    assert _resolve_storage_url(url, tmp_path) == url


def test_sqlite_fallback_has_no_password_and_passes(tmp_path):
    # The local study.db fallback carries no password; it must always resolve.
    resolved = _resolve_storage_url(None, tmp_path)
    assert resolved.startswith("sqlite:///")


def test_error_message_is_actionable(tmp_path):
    # The message must point at the libpq alternatives so a user can fix it.
    with pytest.raises(ValueError) as excinfo:
        _resolve_storage_url(
            "postgresql://user:pw@host/db", tmp_path
        )
    msg = str(excinfo.value)
    assert ".pgpass" in msg
    assert "PGPASSWORD" in msg
    assert "PGSERVICE" in msg


def test_resolved_sqlite_url_round_trips_without_a_password():
    # A sqlite path with a ':' in it (Windows-ish) still has no URL password.
    from urllib.parse import urlsplit

    url = f"sqlite:///{Path('/tmp/a/study.db')}"
    assert urlsplit(url).password is None
