"""Scheme dispatch from a tune storage URL to an Optuna storage object (P1).

Optuna's own string resolver (:func:`optuna.storages.get_storage`) hands every
URL to ``RDBStorage``, which hands it to SQLAlchemy — so there is no hook for a
pseudo-scheme:

.. code-block:: text

    optuna.storages.RDBStorage("journal:///tmp/x/journal.log")
      -> NoSuchModuleError: Can't load plugin: sqlalchemy.dialects:journal

That is why the ``journal://`` backend needs an explicit resolver rather than a
branch alongside ``RDBStorage``, and why every construction site in ``tune/``
must route through :func:`build_optuna_storage` instead of passing the raw URL
string to Optuna.

The lock is :class:`optuna.storages.journal.JournalFileSymlinkLock`, not
``JournalFileOpenLock``: symlink creation is atomic on NFS, while ``O_EXCL``
open semantics are not reliably provided there. On the GPFS mounts this project
runs on, the lock is measurably redundant (POSIX byte-range semantics are
enforced cluster-wide by the token manager) — it ships enabled anyway because it
costs nothing at this workload's timescale and is what makes the same code
correct on an NFS deployment elsewhere.

``import optuna`` stays lazy inside the function bodies, preserving the
package-wide lazy-import boundary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

#: The pseudo-scheme naming a file-backed Optuna ``JournalStorage``. Not a
#: SQLAlchemy dialect — :func:`build_optuna_storage` is what gives it meaning.
JOURNAL_SCHEME: str = "journal"


def is_journal_url(storage_url: Optional[str]) -> bool:
    """Whether ``storage_url`` names the file-backed journal backend.

    Args:
        storage_url: A tune storage URL, or ``None``.

    Returns:
        ``True`` for a ``journal://`` URL, ``False`` for ``None``, a SQLite URL,
        a Postgres URL, or anything else ``RDBStorage`` would take.

    Examples:
        >>> is_journal_url("journal:///runs/out/.pht-tune-cache/journal.log")
        True
        >>> is_journal_url("sqlite:///runs/out/.pht-tune-cache/study.db")
        False
        >>> is_journal_url(None)
        False
    """
    if not storage_url:
        return False
    return urlsplit(storage_url).scheme.split("+", 1)[0] == JOURNAL_SCHEME


def is_sqlite_url(storage_url: Optional[str]) -> bool:
    """Whether ``storage_url`` names a SQLite database.

    Scheme classification only — it says nothing about whether the file exists.
    The ``+``-split matches SQLAlchemy's backend naming, so ``sqlite+pysqlite``
    classifies the same as bare ``sqlite``.

    Args:
        storage_url: A tune storage URL, or ``None``.

    Returns:
        ``True`` for a SQLite URL, ``False`` for ``None`` or any other scheme.

    Examples:
        >>> is_sqlite_url("sqlite:///runs/out/.pht-tune-cache/study.db")
        True
        >>> is_sqlite_url("sqlite+pysqlite:///runs/out/study.db")
        True
        >>> is_sqlite_url("postgresql+psycopg://host/db")
        False
    """
    if not storage_url:
        return False
    return urlsplit(storage_url).scheme.split("+", 1)[0] == "sqlite"


def journal_url_for_path(journal_path: Path) -> str:
    """Return the ``journal://`` URL addressing ``journal_path``.

    Mirrors the ``sqlite:///<path>`` spelling used for the local study DB, so a
    resolved storage URL is a single string that round-trips through
    :func:`journal_path_from_url`.

    Args:
        journal_path: The absolute append-only log path.

    Returns:
        ``journal:///<absolute path>``.

    Examples:
        >>> journal_url_for_path(Path("/runs/out/.pht-tune-cache/journal.log"))
        'journal:///runs/out/.pht-tune-cache/journal.log'
    """
    # Built by hand rather than with ``urlunsplit``: that helper emits the
    # authority separator only for a non-empty netloc, so it would render the
    # single-slash ``journal:/abs/path``. The triple-slash spelling is what
    # ``sqlite:///`` uses and what an operator will type.
    raw = Path(journal_path).as_posix()
    if not raw.startswith("/"):
        raw = f"/{raw}"  # a Windows drive path: C:/x -> journal:///C:/x
    return f"{JOURNAL_SCHEME}://{raw}"


def journal_path_from_url(storage_url: str) -> Path:
    """Return the log path a ``journal://`` URL addresses.

    The inverse of :func:`journal_url_for_path`. A Windows drive path survives
    the round trip: ``urlsplit`` leaves ``journal:///C:/runs/journal.log`` with a
    path of ``/C:/runs/journal.log``, and the leading separator is dropped when
    the next segment is a drive letter.

    Args:
        storage_url: A ``journal://`` storage URL.

    Returns:
        The append-only log path.

    Raises:
        ValueError: When ``storage_url`` is not a ``journal://`` URL, or carries
            no path.

    Examples:
        >>> journal_path_from_url("journal:///runs/out/journal.log").as_posix()
        '/runs/out/journal.log'
    """
    if not is_journal_url(storage_url):
        raise ValueError(
            f"not a {JOURNAL_SCHEME}:// storage URL: {storage_url!r}"
        )
    raw = urlsplit(storage_url).path
    if len(raw) > 2 and raw[0] == "/" and raw[2] == ":":
        raw = raw[1:]  # journal:///C:/... -> C:/...
    if not raw:
        raise ValueError(f"{JOURNAL_SCHEME}:// storage URL carries no path: {storage_url!r}")
    return Path(raw)


def build_optuna_storage(
    storage_url: str,
    *,
    heartbeat_interval: Optional[int] = None,
    grace_period: Optional[int] = None,
) -> Any:
    """Build the Optuna storage object ``storage_url`` names.

    The one dispatch point for the ``journal://`` pseudo-scheme. A
    ``journal://`` URL builds a :class:`optuna.storages.JournalStorage` over a
    symlink-locked :class:`~optuna.storages.journal.JournalFileBackend`; every
    other scheme (``sqlite``, ``postgresql+psycopg``, …) builds the
    ``RDBStorage`` it always did.

    The heartbeat arguments are RDB-only and are **dropped** for the journal
    backend, which implements neither ``get_heartbeat_interval`` nor
    ``record_heartbeat``. That is not a silent downgrade: the strategy layer's
    heartbeat probes are ``getattr``-guarded and degrade to "no heartbeat
    thread" (see ``strategy/_optuna.py``), so the loss is stale-trial
    reclamation, documented in §7 L2 as bounded rather than fatal.

    Args:
        storage_url: The resolved tune storage URL.
        heartbeat_interval: Seconds between worker heartbeats (RDB only).
        grace_period: Seconds before an unheard-from trial is stale (RDB only).

    Returns:
        A live Optuna storage object.

    Examples:
        >>> import tempfile
        >>> tmp = Path(tempfile.mkdtemp())
        >>> storage = build_optuna_storage(journal_url_for_path(tmp / "journal.log"))
        >>> type(storage).__name__
        'JournalStorage'
        >>> storage = build_optuna_storage(f"sqlite:///{tmp / 'study.db'}")
        >>> type(storage).__name__
        'RDBStorage'
    """
    import optuna

    if is_journal_url(storage_url):
        from optuna.storages.journal import (
            JournalFileBackend,
            JournalFileSymlinkLock,
        )

        journal_path = journal_path_from_url(storage_url)
        # The backend does not create parent directories; a `--slurm` submission
        # resolves this URL under `.pht-tune-cache/`, which may not exist yet.
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        text_path = str(journal_path)
        return optuna.storages.JournalStorage(
            JournalFileBackend(
                text_path, lock_obj=JournalFileSymlinkLock(text_path)
            )
        )

    return optuna.storages.RDBStorage(
        url=storage_url,
        heartbeat_interval=heartbeat_interval,
        grace_period=grace_period,
    )
