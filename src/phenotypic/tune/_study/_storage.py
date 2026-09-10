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

import json
import logging
import os
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any, Optional
from urllib.parse import quote, unquote, urlsplit

_logger = logging.getLogger(__name__)

#: The pseudo-scheme naming a file-backed Optuna ``JournalStorage``. Not a
#: SQLAlchemy dialect — :func:`build_optuna_storage` is what gives it meaning.
JOURNAL_SCHEME: str = "journal"

#: Exact query marker for the canonical percent-encoded URL grammar. Queries
#: were invalid in the legacy grammar, so this marker cannot collide with any
#: previously valid journal path identity.
_JOURNAL_CANONICAL_QUERY: str = "v=1"

#: Bytes read per step when scanning the log's tail backwards for the last
#: newline. A journal record is a few hundred bytes, so one chunk finds it in
#: every real case; the loop exists only so a pathologically long record cannot
#: make the repair miss.
_TAIL_SCAN_CHUNK: int = 64 * 1024


def _base_scheme(storage_url: Optional[str]) -> str:
    """Return a URL backend name without its SQLAlchemy driver suffix."""
    if not storage_url:
        return ""
    return urlsplit(storage_url).scheme.split("+", 1)[0]


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
    return _base_scheme(storage_url) == JOURNAL_SCHEME


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
    return _base_scheme(storage_url) == "sqlite"


def journal_url_for_path(journal_path: PurePath) -> str:
    """Return the ``journal://`` URL addressing an **absolute** ``journal_path``.

    Mirrors the ``sqlite:///<path>`` spelling used for the local study DB, so a
    resolved storage URL is a single string that round-trips through
    :func:`journal_path_from_url`. Filesystem path data is percent-encoded; the
    inverse decodes it exactly once. The exact ``?v=1`` query marks this canonical
    grammar; unmarked legacy URLs preserve percent text literally. Windows path
    objects normalize their separators to ``/`` before encoding. A literal
    backslash filename component under POSIX is rejected because its decoded
    identity would become a separator when another worker runs Windows.

    A relative path is **rejected**, not silently rooted. The URL grammar has
    nowhere to put "relative to the caller's cwd", and a fleet resolves this URL
    on nodes whose cwd is not the submitter's — so the only two readings of
    ``out/journal.log`` are both wrong. Rooting it silently is the worse one:
    ``tmp/run1/journal.log`` would become ``/tmp/run1/journal.log``, a path that
    exists, is writable, and is **node-local**, so every worker would open a
    private study and the fleet would never share one (see
    ``_default_journal_url``, which absolutizes before calling this). Callers
    hold the cwd context; this function does not.

    Args:
        journal_path: The absolute append-only log path.

    Returns:
        ``journal:///<percent-encoded absolute path>?v=1``.

    Raises:
        ValueError: When the path is relative or contains a literal backslash.

    Examples:
        >>> journal_url_for_path(Path("/runs/out/.pht-tune-cache/journal.log"))
        'journal:///runs/out/.pht-tune-cache/journal.log?v=1'
        >>> journal_url_for_path(Path("out/journal.log"))
        Traceback (most recent call last):
        ...
        ValueError: journal:// storage path must be absolute: 'out/journal.log'
    """
    # Built by hand rather than with ``urlunsplit``: that helper emits the
    # authority separator only for a non-empty netloc, so it would render the
    # single-slash ``journal:/abs/path``. The triple-slash spelling is what
    # ``sqlite:///`` uses and what an operator will type.
    path = journal_path if isinstance(journal_path, PurePath) else Path(journal_path)
    raw = path.as_posix()
    if not path.is_absolute():
        # ``Path.is_absolute()`` is platform-dependent. Interpret a drive-rooted
        # spelling with Windows semantics without making the host's concrete
        # ``Path`` pretend to have a different filesystem flavor.
        windows_path = PureWindowsPath(raw)
        if not windows_path.is_absolute():
            raise ValueError(
                f"{JOURNAL_SCHEME}:// storage path must be absolute: {raw!r}"
            )
        raw = windows_path.as_posix()
    if "\\" in raw:
        raise ValueError(
            f"{JOURNAL_SCHEME}:// canonical storage path cannot contain a literal"
            f" backslash: {raw!r}"
        )
    if len(raw) > 1 and raw[1] == ":":
        raw = f"/{raw}"  # a Windows drive path: C:/x -> journal:///C:/x
    return f"{JOURNAL_SCHEME}://{quote(raw, safe='/:')}?{_JOURNAL_CANONICAL_QUERY}"


def _decode_canonical_journal_path(encoded_path: str, storage_url: str) -> str:
    """Decode a versioned path only when it has one canonical URL spelling."""
    encoded_path_casefold = encoded_path.casefold()
    if "%2f" in encoded_path_casefold or "%5c" in encoded_path_casefold:
        raise ValueError(
            f"{JOURNAL_SCHEME}:// URL path is not canonical: {storage_url!r}"
        )

    try:
        decoded_path = unquote(encoded_path, encoding="utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{JOURNAL_SCHEME}:// URL path is not canonical: {storage_url!r}"
        ) from exc

    if quote(decoded_path, safe="/:") != encoded_path:
        raise ValueError(
            f"{JOURNAL_SCHEME}:// URL path is not canonical: {storage_url!r}"
        )
    return decoded_path


def journal_path_from_url(storage_url: str) -> Path:
    """Return the log path a ``journal://`` URL addresses.

    The inverse of :func:`journal_url_for_path`. Versioned ``?v=1`` URLs use a
    strict canonical percent grammar and decode once. Unmarked legacy URLs
    preserve percent text literally, including malformed escapes. Encoded
    forward slashes and backslashes are always noncanonical separators. A Windows
    drive path survives the round trip: ``urlsplit`` leaves
    ``journal:///C:/runs/journal.log`` with a path of ``/C:/runs/journal.log``;
    the leading separator is dropped when the next segment is a drive letter.

    A non-empty authority is **refused** rather than dropped. ``urlsplit`` reads
    ``journal://mydir/journal.log`` as netloc ``mydir`` + path
    ``/journal.log``, so silently taking the path would resolve a two-slash typo
    (or a hand-written URL that meant a relative directory) to the filesystem
    root — a different, probably unwritable, and definitely unshared journal.

    Args:
        storage_url: A ``journal://`` storage URL.

    Returns:
        The append-only log path.

    Raises:
        ValueError: When the URL has the wrong scheme, authority, metadata,
            path, version marker, or canonical encoding.

    Examples:
        >>> journal_path_from_url("journal:///runs/out/journal.log").as_posix()
        '/runs/out/journal.log'
        >>> journal_path_from_url("journal://mydir/j.log")
        Traceback (most recent call last):
        ...
        ValueError: journal:// URL must be journal:///<abs path>, got 'journal://mydir/j.log'
    """
    if not is_journal_url(storage_url):
        raise ValueError(
            f"not a {JOURNAL_SCHEME}:// storage URL: {storage_url!r}"
        )
    split = urlsplit(storage_url)
    if split.netloc:
        raise ValueError(
            f"{JOURNAL_SCHEME}:// URL must be {JOURNAL_SCHEME}:///<abs path>"
            f", got {storage_url!r}"
        )
    if "#" in storage_url:
        raise ValueError(
            f"{JOURNAL_SCHEME}:// URL must not carry a query or fragment"
            f", got {storage_url!r}"
        )
    has_query_delimiter = "?" in storage_url
    if has_query_delimiter and split.query != _JOURNAL_CANONICAL_QUERY:
        raise ValueError(
            f"{JOURNAL_SCHEME}:// URL has an unknown version marker or forbidden"
            " query or fragment"
            f", got {storage_url!r}"
        )
    if has_query_delimiter:
        canonical_url = (
            f"{JOURNAL_SCHEME}://{split.path}?{_JOURNAL_CANONICAL_QUERY}"
        )
        if storage_url != canonical_url:
            raise ValueError(
                f"{JOURNAL_SCHEME}:// URL has a noncanonical version marker"
                f", got {storage_url!r}"
            )
        raw = _decode_canonical_journal_path(split.path, storage_url)
    else:
        raw = split.path
    if len(raw) > 2 and raw[0] == "/" and raw[2] == ":":
        raw = raw[1:]  # journal:///C:/... -> C:/...
    if not raw:
        raise ValueError(f"{JOURNAL_SCHEME}:// storage URL carries no path: {storage_url!r}")
    return Path(raw)


def _required_optuna_rdb_tables() -> frozenset[str]:
    """Return Optuna's complete current RDB catalog plus Alembic authority."""
    from optuna.storages._rdb import models

    return frozenset((*models.BaseModel.metadata.tables, "alembic_version"))


def _sqlite_database_path(storage_url: str) -> Path:
    """Return the filesystem database named by a non-memory SQLite URL."""
    from sqlalchemy.engine import make_url

    database = make_url(storage_url).database
    if not database or database == ":memory:":
        raise FileNotFoundError(
            "create=False requires an existing file-backed SQLite database"
        )
    return Path(database)


def require_existing_optuna_storage(storage_url: str) -> None:
    """Validate backing storage without invoking Optuna's creating constructor.

    Journal and SQLite paths are checked before opening them. RDB catalogs must
    contain Optuna's complete model metadata plus populated version-info and
    Alembic authority. Only then may callers construct ``RDBStorage`` with
    table creation disabled.

    Args:
        storage_url: The resolved tune storage URL.

    Raises:
        FileNotFoundError: If a journal or SQLite backing file is absent.
        RuntimeError: If an RDB catalog lacks a complete initialized Optuna schema.
    """
    if is_journal_url(storage_url):
        journal_path = journal_path_from_url(storage_url)
        if not journal_path.is_file():
            raise FileNotFoundError(journal_path)
        return

    if is_sqlite_url(storage_url):
        database_path = _sqlite_database_path(storage_url)
        if not database_path.is_file():
            raise FileNotFoundError(database_path)

    from sqlalchemy import create_engine, inspect
    from sqlalchemy.exc import SQLAlchemyError

    engine = create_engine(storage_url)
    try:
        tables = set(inspect(engine).get_table_names())
        missing = _required_optuna_rdb_tables() - tables
        if missing:
            raise RuntimeError(
                "Optuna RDB schema is not initialized; missing tables: "
                + ", ".join(sorted(missing))
            )
        with engine.connect() as connection:
            version_info = connection.exec_driver_sql(
                "SELECT schema_version, library_version FROM version_info"
            ).first()
            alembic_version = connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).first()
    except SQLAlchemyError as exc:
        raise RuntimeError("Optuna RDB schema version authority is invalid") from exc
    finally:
        engine.dispose()

    missing_authority = []
    if version_info is None:
        missing_authority.append("version_info row")
    if alembic_version is None:
        missing_authority.append("alembic_version row")
    if missing_authority:
        raise RuntimeError(
            "Optuna RDB schema is not initialized; missing authority: "
            + ", ".join(missing_authority)
        )


def truncate_torn_journal_tail(journal_path: Path) -> int:
    """Drop a newline-less trailing record from ``journal_path``; return its size.

    The repair that makes the bounded append retry
    (``retry_on_transient_db_error``) safe on a shared filesystem. A GPFS/NFS
    ``EIO`` inside ``JournalFileBackend.append_logs`` can fail *after* some of
    the record's bytes have landed, leaving the log ending mid-record with no
    terminating newline. Optuna's reader tolerates that **only while the torn
    record is still last** — ``read_logs`` stashes the decode error and raises it
    on the *next* iteration — so the very next append joins the stump to a
    healthy record, turning it into a newline-terminated line of invalid JSON in
    the *middle* of the log. From then on every reader raises, including
    ``create_study``/``load_study``'s own ``_sync_with_backend``, and the shared
    study is unrecoverable for the whole fleet.

    So the retry must repair before it re-appends, not reason that the stump is
    harmless. Truncating back to the last newline is safe for the **trailing**
    record and for no other:

    * an unterminated trailing record was never acknowledged to anybody — the
      ``append_logs`` that would have returned raised instead;
    * ``read_logs`` deletes the offset entry for a bad line
      (``del self._log_number_offset[log_number + 1]``), so no live reader holds
      a byte offset into it — which is exactly why *general* compaction is still
      impossible (see ``_JOURNAL_SIZE_WARN_BYTES``): every other record in the
      file **is** addressed by offset;
    * callers hold the journal lock across repair-then-append, so a tail without
      a newline cannot be a write in progress. It is always a dead stump.

    A log with no newline at all is one unterminated record, so it truncates to
    empty.

    Args:
        journal_path: The append-only log to repair. A missing file is a no-op
            (the append that follows will create it).

    Returns:
        The number of bytes discarded — ``0`` when the log was already intact.
    """
    try:
        handle = open(journal_path, "r+b")
    except FileNotFoundError:  # the append about to run re-creates it
        return 0
    with handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size == 0:
            return 0
        handle.seek(size - 1)
        if handle.read(1) == b"\n":
            return 0
        position = size
        while position > 0:
            start = max(0, position - _TAIL_SCAN_CHUNK)
            handle.seek(start)
            chunk = handle.read(position - start)
            index = chunk.rfind(b"\n")
            if index != -1:
                keep = start + index + 1
                handle.truncate(keep)
                return size - keep
            position = start
        handle.truncate(0)
        return size


#: Cache for :func:`_repairing_journal_backend`'s lazily-built subclass — the
#: base class cannot be named until ``optuna`` is imported, and the import stays
#: inside function bodies (the lazy-import boundary).
_REPAIRING_BACKEND: Optional[type] = None


def _repairing_journal_backend() -> type:
    """The ``JournalFileBackend`` subclass that repairs a torn tail before it appends.

    The repair belongs **inside** ``append_logs``' own lock acquisition, so it
    must override the method rather than wrap it: the symlink lock is not
    reentrant, and re-acquiring it would block for the 30 s grace period and then
    forcibly steal the lock from itself. The append itself is therefore the
    minimal upstream write sequence rather than a delegated call; its required
    compatibility behavior is pinned by
    ``test_repairing_backend_appends_a_readable_record_after_a_torn_tail``.

    Placing it here, rather than in the retry wrapper, covers every writer of the
    log with one edit — the retried ``ask``/``tell``, the pruning channel's
    ``report``, and any worker that joins a fleet whose log a *dead* peer left
    torn — and closes the window a repair outside the lock would leave open.

    Returns:
        The backend class :func:`build_optuna_storage` instantiates.
    """
    global _REPAIRING_BACKEND
    if _REPAIRING_BACKEND is not None:
        return _REPAIRING_BACKEND

    from optuna.storages.journal import JournalFileBackend
    from optuna.storages.journal._file import get_lock_file

    class _RepairingJournalFileBackend(JournalFileBackend):  # type: ignore[misc, valid-type]
        """``JournalFileBackend`` that heals a torn tail before every append."""

        def append_logs(self, logs: list[dict[str, Any]]) -> None:
            with get_lock_file(self._lock):
                discarded = truncate_torn_journal_tail(Path(self._file_path))
                if discarded:
                    _logger.warning(
                        "discarded a %d-byte unterminated record at the end of "
                        "%s before appending; a previous append failed partway "
                        "through and the trial it carried was never recorded.",
                        discarded,
                        self._file_path,
                    )
                payload = (
                    "\n".join(
                        json.dumps(log, separators=(",", ":")) for log in logs
                    )
                    + "\n"
                )
                with open(self._file_path, "ab") as handle:
                    handle.write(payload.encode("utf-8"))
                    handle.flush()
                    os.fsync(handle.fileno())

    _REPAIRING_BACKEND = _RepairingJournalFileBackend
    return _REPAIRING_BACKEND


def build_optuna_storage(
    storage_url: str,
    *,
    heartbeat_interval: Optional[int] = None,
    grace_period: Optional[int] = None,
    create: bool = True,
) -> Any:
    """Build the Optuna storage object ``storage_url`` names.

    The one dispatch point for the ``journal://`` pseudo-scheme. A
    ``journal://`` URL builds a :class:`optuna.storages.JournalStorage` over a
    symlink-locked :class:`~optuna.storages.journal.JournalFileBackend`; every
    other scheme (``sqlite``, ``postgresql+psycopg``, …) builds the
    ``RDBStorage`` it always did.

    The heartbeat arguments are RDB-only and are **dropped** for the journal
    backend, which implements neither ``get_heartbeat_interval`` nor
    ``record_heartbeat``. The strategy layer's heartbeat probes are
    ``getattr``-guarded and degrade to "no heartbeat thread" (see
    ``strategy/_optuna.py``), so nothing crashes — but the loss of stale-trial
    reclamation is **permanent, not transient**, and worth stating plainly.

    With no heartbeat to read, :func:`optuna.storages.fail_stale_trials` has
    nothing to act on: against a ``JournalStorage`` it returns cleanly and changes
    no state, with nothing raised or logged to say it did nothing (verified,
    optuna 4.9.0 — the lone ``ExperimentalWarning`` it emits is the generic
    API-stability notice the RDB path emits too). So a trial left
    ``RUNNING`` by a worker the scheduler killed stays ``RUNNING`` for the life
    of the study — a zombie nothing ever reclaims. Under an RDB the same trial
    is transitioned to ``FAIL`` once its grace period lapses, which is a
    self-healing failure; the journal backend converts it into a standing one,
    on exactly the path (SLURM walltime kills) that produces it.

    What that costs, precisely: the zombie is **excluded** from winner
    selection and from the budget gate, both of which read
    ``terminal_trials()`` / ``completed_count()`` rather than the raw list, so
    the fleet still drains its ``--n-trials`` and can never elect a resultless
    trial. What remains is a row that accumulates in the study and inflates the
    raw ``store.trials`` / ``len(store)`` view — the count the GUI Monitor
    renders and the "still in flight" figure ``_finalize`` warns with, which on
    this backend cannot be told apart from a live worker's trial and therefore
    may never fall to zero. Where an accurate in-flight count matters, use
    Postgres.

    Args:
        storage_url: The resolved tune storage URL.
        heartbeat_interval: Seconds between worker heartbeats (RDB only).
        grace_period: Seconds before an unheard-from trial is stale (RDB only).
        create: Whether constructing a missing backing file or RDB schema is
            permitted.

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

    if not create:
        require_existing_optuna_storage(storage_url)

    if is_journal_url(storage_url):
        from optuna.storages.journal import JournalFileSymlinkLock

        journal_path = journal_path_from_url(storage_url)
        # The backend does not create parent directories; a `--slurm` submission
        # resolves this URL under `.pht-tune-cache/`, which may not exist yet.
        if create:
            journal_path.parent.mkdir(parents=True, exist_ok=True)
        text_path = str(journal_path)
        # `_repairing_journal_backend`, not the stock `JournalFileBackend`: a
        # partially-landed append must be truncated back to the last newline
        # before the next one, or the retry that survives it destroys the study
        # for the whole fleet (see `truncate_torn_journal_tail`).
        backend_cls = _repairing_journal_backend()
        return optuna.storages.JournalStorage(
            backend_cls(text_path, lock_obj=JournalFileSymlinkLock(text_path))
        )

    return optuna.storages.RDBStorage(
        url=storage_url,
        heartbeat_interval=heartbeat_interval,
        grace_period=grace_period,
        skip_table_creation=not create,
    )
