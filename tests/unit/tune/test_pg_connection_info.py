"""``read_pg_connection_info`` — parse the HPCC postgres_server handshake files.

Parses ``connection_info.txt`` (Node FQDN + Port + Superuser) and
``pgpassword.txt`` into a ``postgresql+psycopg://USER:PW@HOST:PORT/DB`` URL,
URL-encoding the password. Missing files raise a clear error and the password is
never echoed.
"""
from __future__ import annotations

import pytest

from phenotypic.tune._study._pg import read_pg_connection_info

#: Mirrors the real ``connection_info.txt`` written by ``pgserver.sh``.
_CONNECTION_INFO = """\
PostgreSQL server — connection info        (written Wed Jun  4 10:00:00 2026)
================================================================
Job ID      : 12345
Node        : i12   (i12.ib.hpcc.ucr.edu)
Node IP     : 10.1.2.3
Port        : 54399
Data dir    : /rhome/alice/util/postgres_server/pgdata
Superuser   : alice
Password    : stored in /rhome/alice/util/postgres_server/pgpassword.txt

Connect — SAME node (Unix socket, no password):
    psql -h /rhome/alice/util/postgres_server/pgdata -p 54399 -U alice postgres
================================================================
"""


def _server_dir(tmp_path, *, password: str = "s3cr3t"):
    d = tmp_path / "postgres_server"
    d.mkdir()
    (d / "connection_info.txt").write_text(_CONNECTION_INFO)
    (d / "pgpassword.txt").write_text(password + "\n")
    return d


def test_builds_psycopg_url_from_fixture(tmp_path):
    d = _server_dir(tmp_path)
    url = read_pg_connection_info(server_dir=d, db="tune")
    assert url == (
        "postgresql+psycopg://alice:s3cr3t@i12.ib.hpcc.ucr.edu:54399/tune"
    )


def test_password_is_url_encoded(tmp_path):
    # A password with URL-reserved characters must be percent-encoded so the URL
    # stays well-formed.
    from urllib.parse import unquote

    d = _server_dir(tmp_path, password="p@ss/w:rd#1")
    url = read_pg_connection_info(server_dir=d, db="tune")
    assert "p@ss/w:rd#1" not in url  # the raw, reserved-char password is gone
    assert url.startswith("postgresql+psycopg://alice:")
    assert url.endswith("@i12.ib.hpcc.ucr.edu:54399/tune")
    # The encoded password round-trips back to the original.
    encoded = url.split("alice:", 1)[1].split("@", 1)[0]
    assert unquote(encoded) == "p@ss/w:rd#1"


def test_uses_port_54399(tmp_path):
    d = _server_dir(tmp_path)
    url = read_pg_connection_info(server_dir=d, db="x")
    assert ":54399/" in url


def test_missing_connection_info_raises_clear_error(tmp_path):
    d = tmp_path / "postgres_server"
    d.mkdir()
    (d / "pgpassword.txt").write_text("pw\n")
    with pytest.raises(FileNotFoundError) as exc:
        read_pg_connection_info(server_dir=d, db="tune")
    assert "connection_info.txt" in str(exc.value)


def test_missing_password_file_raises_clear_error(tmp_path):
    d = tmp_path / "postgres_server"
    d.mkdir()
    (d / "connection_info.txt").write_text(_CONNECTION_INFO)
    with pytest.raises(FileNotFoundError) as exc:
        read_pg_connection_info(server_dir=d, db="tune")
    assert "pgpassword.txt" in str(exc.value)


def test_error_does_not_leak_password(tmp_path):
    # If parsing the connection_info fails (e.g. no Node line), the raised error
    # must not contain the password contents.
    d = tmp_path / "postgres_server"
    d.mkdir()
    (d / "connection_info.txt").write_text("garbage with no fields\n")
    (d / "pgpassword.txt").write_text("TOPSECRET\n")
    with pytest.raises(ValueError) as exc:
        read_pg_connection_info(server_dir=d, db="tune")
    assert "TOPSECRET" not in str(exc.value)
