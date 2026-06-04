"""Build a Postgres storage URL from the HPCC ``postgres_server`` handshake files.

The user-space Postgres server (``~/util/postgres_server/pgserver.sh``) writes
its current address to ``connection_info.txt`` on every launch (the Slurm node
changes each run) and a superuser password to ``pgpassword.txt`` (mode 600).
:func:`read_pg_connection_info` parses both into a SQLAlchemy/psycopg URL —
``postgresql+psycopg://USER:PW@HOST:PORT/DB`` — that an Optuna ``RDBStorage`` (or
the distributed study handles) can open. The password is URL-encoded and never
logged or echoed into raised errors.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

#: The canonical server folder the HPCC launcher writes its handshake files to.
DEFAULT_PG_SERVER_DIR: Path = Path.home() / "util" / "postgres_server"

#: The launcher's high port (``PORT=54399`` in ``pgserver.sh``). Parsed from the
#: file, but kept here as the documented default the server is configured with.
PG_DEFAULT_PORT: int = 54399

_CONNECTION_INFO_FILE = "connection_info.txt"
_PASSWORD_FILE = "pgpassword.txt"

#: ``Node        : i12   (i12.ib.hpcc.ucr.edu)`` — capture the FQDN in parens
#: (preferred, it resolves cross-node) or the bare short name when no FQDN.
_NODE_RE = re.compile(r"^Node\s*:\s*(?P<short>\S+)(?:\s*\((?P<fqdn>[^)]+)\))?", re.M)
#: ``Port        : 54399``
_PORT_RE = re.compile(r"^Port\s*:\s*(?P<port>\d+)", re.M)
#: ``Superuser   : alice``
_USER_RE = re.compile(r"^Superuser\s*:\s*(?P<user>\S+)", re.M)


def read_pg_connection_info(
    server_dir: Path = DEFAULT_PG_SERVER_DIR, *, db: str
) -> str:
    """Build a ``postgresql+psycopg://`` URL from the server handshake files.

    Reads ``<server_dir>/connection_info.txt`` for the host (FQDN preferred),
    port, and superuser, and ``<server_dir>/pgpassword.txt`` for the password,
    then assembles ``postgresql+psycopg://USER:PW@HOST:PORT/DB`` with the
    password URL-encoded.

    Args:
        server_dir: The ``postgres_server`` folder holding the handshake files
            (defaults to ``~/util/postgres_server``).
        db: The database name to connect to (the path component of the URL).

    Returns:
        The psycopg storage URL for ``db`` on the running server.

    Raises:
        FileNotFoundError: If ``connection_info.txt`` or ``pgpassword.txt`` is
            absent (e.g. the server is not running) — the message names the
            missing file, never the password.
        ValueError: If ``connection_info.txt`` lacks a recognizable Node / Port
            / Superuser line — the message never includes the password.
    """
    server_dir = Path(server_dir)
    info_path = server_dir / _CONNECTION_INFO_FILE
    pw_path = server_dir / _PASSWORD_FILE

    if not info_path.exists():
        raise FileNotFoundError(
            f"Postgres connection_info.txt not found at {info_path}. "
            "Is the postgres_server Slurm job running?"
        )
    if not pw_path.exists():
        raise FileNotFoundError(
            f"Postgres pgpassword.txt not found at {pw_path}. "
            "Run pgserver.sh once to initialize the database password."
        )

    info = info_path.read_text()
    node_match = _NODE_RE.search(info)
    port_match = _PORT_RE.search(info)
    user_match = _USER_RE.search(info)
    if node_match is None or port_match is None or user_match is None:
        # Do NOT include the file contents or password in the error.
        raise ValueError(
            f"Could not parse host/port/user from {info_path}; the file is not "
            "in the expected pgserver.sh connection_info.txt format."
        )

    host = node_match.group("fqdn") or node_match.group("short")
    port = port_match.group("port")
    user = user_match.group("user")
    # The password file may hold a trailing newline; strip surrounding whitespace.
    password = pw_path.read_text().strip()

    user_enc = quote(user, safe="")
    password_enc = quote(password, safe="")
    return f"postgresql+psycopg://{user_enc}:{password_enc}@{host}:{port}/{db}"
