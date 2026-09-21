"""Handle-bound Windows primitives for identity-bound directory I/O.

This module owns the native Win32/NT plumbing: every child is opened
relative to a held directory handle via ``NtCreateFile``, identity is read
from ``FILE_ID_INFO``, and reparse points are visible in handle metadata
rather than followed. It carries no journal, receipt or transaction
semantics -- :mod:`phenotypic.sdk_._windows_metadata_journal` imports these
names and layers its own semantics on top.

It also carries the Windows backend of the identity-IO facade
(:mod:`phenotypic.sdk_._identity_io`): :class:`_WindowsHeldDirectory` and
:func:`open_identity_directory`, which satisfy the same refusal contract the
POSIX backend does.

:class:`WindowsJournalUnavailable` lives here rather than with the journal
because :class:`_CtypesWindowsApi` raises it, and a module that raises an
exception defined by its own importer is a cycle.
"""

from __future__ import annotations

import ctypes
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Protocol

BACKEND_NAME = "windows"


class WindowsJournalUnavailable(RuntimeError):
    """Raised before mutation when the required Windows handle API is absent."""


@dataclass(frozen=True)
class WindowsHandleInfo:
    """Stable identity and no-follow attributes for one Windows handle."""

    volume_serial: int
    file_id: bytes
    attributes: int
    reparse_tag: int

    @property
    def identity(self) -> tuple[int, bytes]:
        """Return the volume-scoped file identity."""
        return self.volume_serial, self.file_id


class _WindowsApi(Protocol):
    def open_anchor(self, anchor: str, *, share_delete: bool) -> int: ...

    def open_directory(
        self,
        parent: int,
        name: str,
        *,
        create: bool,
        share_delete: bool,
    ) -> int: ...

    def open_file(
        self,
        parent: int,
        name: str,
        *,
        create_new: bool,
        share_delete: bool,
    ) -> int: ...

    def handle_info(self, handle: int) -> WindowsHandleInfo: ...

    def write_all(self, handle: int, payload: bytes) -> None: ...

    def read_all(self, handle: int) -> bytes: ...

    def flush(self, handle: int) -> None: ...

    def rename(
        self,
        handle: int,
        parent: int,
        name: str,
        *,
        replace: bool,
    ) -> None: ...

    def delete(self, handle: int) -> None: ...

    def close(self, handle: int) -> None: ...

    def lock(self, handle: int) -> Any: ...


class _IdentityWindowsApi(Protocol):
    """The read-only subset :class:`_WindowsHeldDirectory` drives.

    Deliberately not :class:`_WindowsApi`: the journal's protocol describes a
    *writer* (``write_all``, ``rename``, ``delete``, ``lock``), none of which
    this backend may call, and its ``open_file`` asks for
    ``FILE_WRITE_DATA | FILE_WRITE_ATTRIBUTES | DELETE`` -- a mask a read-only
    store or a deny-write ACL refuses outright. A separate protocol keeps the
    test fake for one from satisfying the other by accident.
    """

    def open_anchor(self, anchor: str, *, share_delete: bool) -> int: ...

    def open_directory(
        self,
        parent: int,
        name: str,
        *,
        create: bool,
        share_delete: bool,
    ) -> int: ...

    def open_regular_read(self, parent: int, name: str) -> int: ...

    def handle_info(self, handle: int) -> WindowsHandleInfo: ...

    def is_directory(self, handle: int) -> bool: ...

    def link_count(self, handle: int) -> int: ...

    def list_names(self, handle: int) -> tuple[str, ...]: ...

    def adopt_descriptor(self, handle: int) -> int: ...

    def stream(self, handle: int) -> BinaryIO: ...

    def close(self, handle: int) -> None: ...


# Win32/NT constants used only by the ctypes adapter.
_DELETE = 0x00010000
_SYNCHRONIZE = 0x00100000
_FILE_READ_DATA = 0x0001
_FILE_WRITE_DATA = 0x0002
_FILE_LIST_DIRECTORY = 0x0001
_FILE_TRAVERSE = 0x0020
_FILE_READ_ATTRIBUTES = 0x0080
_FILE_WRITE_ATTRIBUTES = 0x0100
_FILE_SHARE_READ = 0x0001
_FILE_SHARE_WRITE = 0x0002
_FILE_OPEN = 0x00000001
_FILE_CREATE = 0x00000002
_FILE_OPEN_IF = 0x00000003
_FILE_DIRECTORY_FILE = 0x00000001
_FILE_WRITE_THROUGH = 0x00000002
_FILE_SYNCHRONOUS_IO_NONALERT = 0x00000020
_FILE_NON_DIRECTORY_FILE = 0x00000040
_FILE_OPEN_REPARSE_POINT = 0x00200000
_FILE_ATTRIBUTE_NORMAL = 0x00000080
#: ``FILE_ATTRIBUTE_REPARSE_POINT``. Lives here rather than with the journal
#: because both the journal and this module's own backend test it, and the
#: journal already imports its Win32 vocabulary from here.
_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_OPEN_EXISTING = 3
_FILE_STANDARD_INFO_CLASS = 1
_FILE_ATTRIBUTE_TAG_INFO_CLASS = 9
_FILE_FULL_DIRECTORY_INFO_CLASS = 14
_FILE_FULL_DIRECTORY_RESTART_INFO_CLASS = 15
_FILE_ID_INFO_CLASS = 18
#: ``FILE_INFORMATION_CLASS.FileRenameInformation`` for ``NtSetInformationFile``.
#: Not the Win32 ``FileRenameInfo`` (3): ``SetFileInformationByHandle`` rejects a
#: rename relative to ``RootDirectory`` with ``ERROR_INVALID_PARAMETER``, which is
#: the only form that keeps the target bound to the held parent handle.
_FILE_RENAME_INFORMATION_CLASS = 10
_FILE_DISPOSITION_INFO_CLASS = 4
_OBJ_CASE_INSENSITIVE = 0x40
_LOCKFILE_EXCLUSIVE_LOCK = 0x2
_LOCKFILE_FAIL_IMMEDIATELY = 0x1
_ERROR_ALREADY_EXISTS = 183
_ERROR_FILE_EXISTS = 80
_ERROR_LOCK_VIOLATION = 33
#: ``RtlNtStatusToDosError(STATUS_FILE_IS_A_DIRECTORY)`` folds onto the same
#: code a real ACL denial produces, so 5 alone never proves a type mismatch;
#: :meth:`_WindowsHeldDirectory._refuse_type_mismatch` disambiguates it.
_ERROR_ACCESS_DENIED = 5
#: ``RtlNtStatusToDosError(STATUS_NOT_A_DIRECTORY)`` -- ``FILE_DIRECTORY_FILE``
#: against a regular file.
_ERROR_DIRECTORY = 267
_ERROR_NO_MORE_FILES = 18

#: One ``GetFileInformationByHandleEx`` fill. A directory larger than this is
#: read across several calls; ``_ERROR_NO_MORE_FILES`` ends the walk.
_LISTING_BUFFER_BYTES = 64 * 1024

#: ``os.O_BINARY`` exists only on Windows, and this module is imported on every
#: platform (the journal re-exports from it). Zero is the correct no-op
#: elsewhere, where the descriptor is already binary.
_O_BINARY = getattr(os, "O_BINARY", 0)


class _UnicodeString(ctypes.Structure):
    _fields_ = [
        ("Length", ctypes.c_uint16),
        ("MaximumLength", ctypes.c_uint16),
        ("Buffer", ctypes.c_wchar_p),
    ]


class _ObjectAttributes(ctypes.Structure):
    _fields_ = [
        ("Length", ctypes.c_uint32),
        ("RootDirectory", ctypes.c_void_p),
        ("ObjectName", ctypes.POINTER(_UnicodeString)),
        ("Attributes", ctypes.c_uint32),
        ("SecurityDescriptor", ctypes.c_void_p),
        ("SecurityQualityOfService", ctypes.c_void_p),
    ]


class _IoStatusBlock(ctypes.Structure):
    _fields_ = [("Status", ctypes.c_void_p), ("Information", ctypes.c_size_t)]


class _FileIdInfo(ctypes.Structure):
    _fields_ = [
        ("VolumeSerialNumber", ctypes.c_uint64),
        ("FileId", ctypes.c_ubyte * 16),
    ]


class _FileAttributeTagInfo(ctypes.Structure):
    _fields_ = [("FileAttributes", ctypes.c_uint32), ("ReparseTag", ctypes.c_uint32)]


class _FileStandardInfo(ctypes.Structure):
    """``FILE_STANDARD_INFO``: link count, size and the directory bit.

    The 2026-09-20 probe read this class successfully from both an
    ``NtCreateFile`` directory handle and a file handle, so one call answers
    all three questions this backend asks about an entry.
    """

    _fields_ = [
        ("AllocationSize", ctypes.c_int64),
        ("EndOfFile", ctypes.c_int64),
        ("NumberOfLinks", ctypes.c_uint32),
        ("DeletePending", ctypes.c_ubyte),
        ("Directory", ctypes.c_ubyte),
    ]


class _FileFullDirInfo(ctypes.Structure):
    """``FILE_FULL_DIR_INFO``, the entry layout of a directory listing.

    ``FileName`` is a variable-length trailing array, so only its *offset* is
    used; the name is read with :func:`ctypes.wstring_at` at
    ``offset + FileName.offset`` for ``FileNameLength // 2`` characters. The
    probe measured that offset as 68, which is what this declaration yields.
    """

    _fields_ = [
        ("NextEntryOffset", ctypes.c_uint32),
        ("FileIndex", ctypes.c_uint32),
        ("CreationTime", ctypes.c_int64),
        ("LastAccessTime", ctypes.c_int64),
        ("LastWriteTime", ctypes.c_int64),
        ("ChangeTime", ctypes.c_int64),
        ("EndOfFile", ctypes.c_int64),
        ("AllocationSize", ctypes.c_int64),
        ("FileAttributes", ctypes.c_uint32),
        ("FileNameLength", ctypes.c_uint32),
        ("EaSize", ctypes.c_uint32),
        ("FileName", ctypes.c_uint16 * 1),
    ]


class _FileDispositionInfo(ctypes.Structure):
    _fields_ = [("DeleteFile", ctypes.c_ubyte)]


class _FileRenameInfoHeader(ctypes.Structure):
    _fields_ = [
        ("Flags", ctypes.c_uint32),
        ("RootDirectory", ctypes.c_void_p),
        ("FileNameLength", ctypes.c_uint32),
        ("FileName", ctypes.c_uint16 * 1),
    ]


class _Overlapped(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", ctypes.c_uint32),
        ("OffsetHigh", ctypes.c_uint32),
        ("hEvent", ctypes.c_void_p),
    ]


class _CtypesWindowsApi:
    """Thin ctypes binding over the required documented Win32/NT primitives."""

    def __init__(self) -> None:
        if os.name != "nt" or not hasattr(ctypes, "WinDLL"):
            raise WindowsJournalUnavailable(
                "Windows metadata journal APIs are unavailable on this platform"
            )
        try:
            self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self.ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
            self._bind()
        except (AttributeError, OSError) as exc:
            raise WindowsJournalUnavailable(
                "Required Windows metadata journal APIs are unavailable"
            ) from exc

    def _bind(self) -> None:
        handle = ctypes.c_void_p
        dword = ctypes.c_uint32
        bool_type = ctypes.c_int32
        self.CreateFileW = self.kernel32.CreateFileW
        self.CreateFileW.argtypes = [
            ctypes.c_wchar_p,
            dword,
            dword,
            ctypes.c_void_p,
            dword,
            dword,
            handle,
        ]
        self.CreateFileW.restype = ctypes.c_void_p
        self.GetFileInformationByHandleEx = (
            self.kernel32.GetFileInformationByHandleEx
        )
        self.GetFileInformationByHandleEx.argtypes = [
            handle,
            ctypes.c_int32,
            ctypes.c_void_p,
            dword,
        ]
        self.GetFileInformationByHandleEx.restype = bool_type
        self.SetFileInformationByHandle = self.kernel32.SetFileInformationByHandle
        self.SetFileInformationByHandle.argtypes = [
            handle,
            ctypes.c_int32,
            ctypes.c_void_p,
            dword,
        ]
        self.SetFileInformationByHandle.restype = bool_type
        self.FlushFileBuffers = self.kernel32.FlushFileBuffers
        self.FlushFileBuffers.argtypes = [handle]
        self.FlushFileBuffers.restype = bool_type
        self.ReadFile = self.kernel32.ReadFile
        self.ReadFile.argtypes = [
            handle,
            ctypes.c_void_p,
            dword,
            ctypes.POINTER(dword),
            ctypes.c_void_p,
        ]
        self.ReadFile.restype = bool_type
        self.WriteFile = self.kernel32.WriteFile
        self.WriteFile.argtypes = [
            handle,
            ctypes.c_void_p,
            dword,
            ctypes.POINTER(dword),
            ctypes.c_void_p,
        ]
        self.WriteFile.restype = bool_type
        self.SetFilePointerEx = self.kernel32.SetFilePointerEx
        self.SetFilePointerEx.argtypes = [
            handle,
            ctypes.c_int64,
            ctypes.c_void_p,
            dword,
        ]
        self.SetFilePointerEx.restype = bool_type
        self.GetFileSizeEx = self.kernel32.GetFileSizeEx
        self.GetFileSizeEx.argtypes = [handle, ctypes.POINTER(ctypes.c_int64)]
        self.GetFileSizeEx.restype = bool_type
        self.CloseHandle = self.kernel32.CloseHandle
        self.CloseHandle.argtypes = [handle]
        self.CloseHandle.restype = bool_type
        self.LockFileEx = self.kernel32.LockFileEx
        self.LockFileEx.argtypes = [
            handle,
            dword,
            dword,
            dword,
            dword,
            ctypes.POINTER(_Overlapped),
        ]
        self.LockFileEx.restype = bool_type
        self.UnlockFileEx = self.kernel32.UnlockFileEx
        self.UnlockFileEx.argtypes = [
            handle,
            dword,
            dword,
            dword,
            ctypes.POINTER(_Overlapped),
        ]
        self.UnlockFileEx.restype = bool_type
        self.NtCreateFile = self.ntdll.NtCreateFile
        self.NtCreateFile.argtypes = [
            ctypes.POINTER(handle),
            dword,
            ctypes.POINTER(_ObjectAttributes),
            ctypes.POINTER(_IoStatusBlock),
            ctypes.c_void_p,
            dword,
            dword,
            dword,
            dword,
            ctypes.c_void_p,
            dword,
        ]
        self.NtCreateFile.restype = ctypes.c_int32
        self.NtSetInformationFile = self.ntdll.NtSetInformationFile
        self.NtSetInformationFile.argtypes = [
            handle,
            ctypes.POINTER(_IoStatusBlock),
            ctypes.c_void_p,
            dword,
            ctypes.c_int32,
        ]
        self.NtSetInformationFile.restype = ctypes.c_int32
        self.RtlNtStatusToDosError = self.ntdll.RtlNtStatusToDosError
        self.RtlNtStatusToDosError.argtypes = [ctypes.c_int32]
        self.RtlNtStatusToDosError.restype = ctypes.c_uint32

    @staticmethod
    def _invalid_handle(handle: int | None) -> bool:
        return handle in {None, ctypes.c_void_p(-1).value}

    @staticmethod
    def _raise_last_error(operation: str) -> None:
        get_last_error = getattr(ctypes, "get_last_error", None)
        if get_last_error is None:
            raise WindowsJournalUnavailable(
                "ctypes Windows last-error support is unavailable"
            )
        error = int(get_last_error())
        if error in {_ERROR_ALREADY_EXISTS, _ERROR_FILE_EXISTS}:
            raise FileExistsError(error, operation)
        raise OSError(error, f"{operation} failed")

    def open_anchor(self, anchor: str, *, share_delete: bool) -> int:
        share = _FILE_SHARE_READ | _FILE_SHARE_WRITE
        if share_delete:
            share |= 0x4
        handle = self.CreateFileW(
            anchor,
            _FILE_LIST_DIRECTORY | _FILE_TRAVERSE | _FILE_READ_ATTRIBUTES,
            share,
            None,
            _OPEN_EXISTING,
            _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        if self._invalid_handle(handle):
            self._raise_last_error("CreateFileW anchor open")
        return int(handle)

    def _nt_open(
        self,
        parent: int,
        name: str,
        *,
        desired_access: int,
        disposition: int,
        options: int,
        attributes: int,
        share_delete: bool,
    ) -> int:
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError(f"Unsafe relative Windows journal name: {name!r}")
        name_buffer = ctypes.create_unicode_buffer(name)
        name_bytes = len(name.encode("utf-16-le"))
        unicode_name = _UnicodeString(
            Length=name_bytes,
            MaximumLength=name_bytes + 2,
            Buffer=ctypes.cast(name_buffer, ctypes.c_wchar_p),
        )
        object_attributes = _ObjectAttributes(
            Length=ctypes.sizeof(_ObjectAttributes),
            RootDirectory=ctypes.c_void_p(parent),
            ObjectName=ctypes.pointer(unicode_name),
            Attributes=_OBJ_CASE_INSENSITIVE,
            SecurityDescriptor=None,
            SecurityQualityOfService=None,
        )
        io_status = _IoStatusBlock()
        output = ctypes.c_void_p()
        share = _FILE_SHARE_READ | _FILE_SHARE_WRITE
        if share_delete:
            share |= 0x4
        status = self.NtCreateFile(
            ctypes.byref(output),
            desired_access,
            ctypes.byref(object_attributes),
            ctypes.byref(io_status),
            None,
            attributes,
            share,
            disposition,
            options,
            None,
            0,
        )
        if status < 0:
            error = int(self.RtlNtStatusToDosError(status))
            if error in {_ERROR_ALREADY_EXISTS, _ERROR_FILE_EXISTS}:
                raise FileExistsError(error, name)
            if error in {2, 3}:
                raise FileNotFoundError(error, name)
            raise OSError(error, f"NtCreateFile failed for {name!r}")
        if not output.value:
            raise WindowsJournalUnavailable("NtCreateFile returned no handle")
        return int(output.value)

    def open_directory(
        self,
        parent: int,
        name: str,
        *,
        create: bool,
        share_delete: bool,
    ) -> int:
        return self._nt_open(
            parent,
            name,
            desired_access=(
                _FILE_LIST_DIRECTORY
                | _FILE_TRAVERSE
                | _FILE_READ_ATTRIBUTES
                | _SYNCHRONIZE
            ),
            disposition=_FILE_OPEN_IF if create else _FILE_OPEN,
            options=(
                _FILE_DIRECTORY_FILE
                | _FILE_OPEN_REPARSE_POINT
                | _FILE_SYNCHRONOUS_IO_NONALERT
            ),
            attributes=_FILE_ATTRIBUTE_NORMAL,
            share_delete=share_delete,
        )

    def open_file(
        self,
        parent: int,
        name: str,
        *,
        create_new: bool,
        share_delete: bool,
    ) -> int:
        return self._nt_open(
            parent,
            name,
            desired_access=(
                _FILE_READ_DATA
                | _FILE_WRITE_DATA
                | _FILE_READ_ATTRIBUTES
                | _FILE_WRITE_ATTRIBUTES
                | _DELETE
                | _SYNCHRONIZE
            ),
            disposition=_FILE_CREATE if create_new else _FILE_OPEN,
            options=(
                _FILE_NON_DIRECTORY_FILE
                | _FILE_OPEN_REPARSE_POINT
                | _FILE_SYNCHRONOUS_IO_NONALERT
                | _FILE_WRITE_THROUGH
            ),
            attributes=_FILE_ATTRIBUTE_NORMAL,
            share_delete=share_delete,
        )

    def handle_info(self, handle: int) -> WindowsHandleInfo:
        identity = _FileIdInfo()
        if not self.GetFileInformationByHandleEx(
            handle,
            _FILE_ID_INFO_CLASS,
            ctypes.byref(identity),
            ctypes.sizeof(identity),
        ):
            self._raise_last_error("GetFileInformationByHandleEx(FileIdInfo)")
        attributes = _FileAttributeTagInfo()
        if not self.GetFileInformationByHandleEx(
            handle,
            _FILE_ATTRIBUTE_TAG_INFO_CLASS,
            ctypes.byref(attributes),
            ctypes.sizeof(attributes),
        ):
            self._raise_last_error(
                "GetFileInformationByHandleEx(FileAttributeTagInfo)"
            )
        return WindowsHandleInfo(
            volume_serial=int(identity.VolumeSerialNumber),
            file_id=bytes(identity.FileId),
            attributes=int(attributes.FileAttributes),
            reparse_tag=int(attributes.ReparseTag),
        )

    def _standard_info(self, handle: int) -> _FileStandardInfo:
        buffer = _FileStandardInfo()
        if not self.GetFileInformationByHandleEx(
            handle,
            _FILE_STANDARD_INFO_CLASS,
            ctypes.byref(buffer),
            ctypes.sizeof(buffer),
        ):
            self._raise_last_error(
                "GetFileInformationByHandleEx(FileStandardInfo)"
            )
        return buffer

    def link_count(self, handle: int) -> int:
        """Return ``st_nlink``'s Windows spelling for an open handle."""
        return int(self._standard_info(handle).NumberOfLinks)

    def file_size(self, handle: int) -> int:
        """Return the byte length of the file behind an open handle."""
        return int(self._standard_info(handle).EndOfFile)

    def is_directory(self, handle: int) -> bool:
        """Whether an open handle names a directory.

        ``open_anchor`` goes through ``CreateFileW`` with
        ``FILE_FLAG_BACKUP_SEMANTICS``, which succeeds on a *regular file* --
        the probe confirmed it. POSIX refuses that twice (``O_DIRECTORY`` and
        the ``S_ISDIR`` check), so without this the two backends disagree on
        ``open_identity_directory(<a file>)``.
        """
        return bool(self._standard_info(handle).Directory)

    def open_regular_read(self, parent: int, name: str) -> int:
        """Open a member for reading only, sharing it with every other writer.

        Not :meth:`open_file`: that asks for
        ``FILE_WRITE_DATA | FILE_WRITE_ATTRIBUTES | DELETE`` because the
        journal writes through it, and the probe confirmed that mask is
        refused with ``ERROR_ACCESS_DENIED`` on a read-only-attribute file and
        on one whose ACL denies ``WD,AD``. A viewer must ask for no more than
        it needs, and must never lock a store against a running CLI --
        ``share_delete=True`` plus ``_nt_open``'s read/write sharing is the
        full share mode.
        """
        return self._nt_open(
            parent,
            name,
            desired_access=(
                _FILE_READ_DATA | _FILE_READ_ATTRIBUTES | _SYNCHRONIZE
            ),
            disposition=_FILE_OPEN,
            options=(
                _FILE_NON_DIRECTORY_FILE
                | _FILE_OPEN_REPARSE_POINT
                | _FILE_SYNCHRONOUS_IO_NONALERT
            ),
            attributes=_FILE_ATTRIBUTE_NORMAL,
            share_delete=True,
        )

    def list_names(self, handle: int) -> tuple[str, ...]:
        """Return every raw entry name in a held directory.

        The names are returned exactly as Windows reports them, **including
        ``.`` and ``..``** -- the probe observed
        ``['.', '..', 'alpha.json', 'beta.json']``. Filtering is the held
        directory's job (:meth:`_WindowsHeldDirectory.list_names`), so that one
        implementation of the filter covers every API this backend runs
        against rather than only this one.
        """
        names: list[str] = []
        buffer = ctypes.create_string_buffer(_LISTING_BUFFER_BYTES)
        info_class = _FILE_FULL_DIRECTORY_RESTART_INFO_CLASS
        while self.GetFileInformationByHandleEx(
            handle, info_class, buffer, ctypes.sizeof(buffer)
        ):
            info_class = _FILE_FULL_DIRECTORY_INFO_CLASS
            offset = 0
            while True:
                entry = ctypes.cast(
                    ctypes.byref(buffer, offset),
                    ctypes.POINTER(_FileFullDirInfo),
                ).contents
                names.append(
                    ctypes.wstring_at(
                        ctypes.byref(
                            buffer, offset + _FileFullDirInfo.FileName.offset
                        ),
                        entry.FileNameLength // 2,
                    )
                )
                if entry.NextEntryOffset == 0:
                    break
                offset += entry.NextEntryOffset
        get_last_error = getattr(ctypes, "get_last_error", None)
        if get_last_error is None:
            raise WindowsJournalUnavailable(
                "ctypes Windows last-error support is unavailable"
            )
        error = int(get_last_error())
        if error != _ERROR_NO_MORE_FILES:
            raise OSError(
                error, "GetFileInformationByHandleEx(directory listing) failed"
            )
        return tuple(names)

    def adopt_descriptor(self, handle: int) -> int:
        """Adopt a handle as a CRT descriptor. Ownership transfers to the fd.

        The caller closes with :func:`os.close`; calling ``CloseHandle`` as
        well is a double close. The probe confirmed that :func:`os.fstat` on
        the resulting descriptor agrees exactly with :func:`os.stat` on the
        path -- ``st_ino``, ``st_ctime_ns``, ``st_mtime_ns`` and ``st_size``
        all identical -- which is what makes the two
        ``store_publication_token`` branches agree by construction.
        """
        if sys.platform == "win32":
            import msvcrt

            return msvcrt.open_osfhandle(handle, os.O_RDONLY | _O_BINARY)
        raise WindowsJournalUnavailable(
            "descriptor adoption requires the Windows CRT"
        )

    def stream(self, handle: int) -> BinaryIO:
        """Wrap a handle as a Python binary file; the fd owns it from here.

        The stream is seekable and stays readable after every directory handle
        above it is closed, so the Browse route can hand it to ``send_file``
        and let the hold go.
        """
        return os.fdopen(self.adopt_descriptor(handle), "rb")

    def write_all(self, handle: int, payload: bytes) -> None:
        self.SetFilePointerEx(handle, 0, None, 0)
        offset = 0
        while offset < len(payload):
            chunk = payload[offset : offset + 1024 * 1024]
            buffer = ctypes.create_string_buffer(chunk)
            written = ctypes.c_uint32()
            if not self.WriteFile(
                handle,
                buffer,
                len(chunk),
                ctypes.byref(written),
                None,
            ):
                self._raise_last_error("WriteFile")
            if written.value <= 0:
                raise OSError("WriteFile made no progress")
            offset += int(written.value)

    def read_all(self, handle: int) -> bytes:
        size = ctypes.c_int64()
        if not self.GetFileSizeEx(handle, ctypes.byref(size)):
            self._raise_last_error("GetFileSizeEx")
        if size.value < 0:
            raise OSError("Windows journal file has a negative size")
        if not self.SetFilePointerEx(handle, 0, None, 0):
            self._raise_last_error("SetFilePointerEx")
        remaining = int(size.value)
        result = bytearray()
        while remaining:
            length = min(remaining, 1024 * 1024)
            buffer = ctypes.create_string_buffer(length)
            read = ctypes.c_uint32()
            if not self.ReadFile(
                handle,
                buffer,
                length,
                ctypes.byref(read),
                None,
            ):
                self._raise_last_error("ReadFile")
            if read.value <= 0:
                raise OSError("ReadFile reached an unexpected end of file")
            result.extend(buffer.raw[: read.value])
            remaining -= int(read.value)
        return bytes(result)

    def flush(self, handle: int) -> None:
        if not self.FlushFileBuffers(handle):
            self._raise_last_error("FlushFileBuffers")

    def rename(
        self,
        handle: int,
        parent: int,
        name: str,
        *,
        replace: bool,
    ) -> None:
        encoded = name.encode("utf-16-le")
        total = _FileRenameInfoHeader.FileName.offset + len(encoded)
        buffer = ctypes.create_string_buffer(total)
        header = ctypes.cast(
            buffer, ctypes.POINTER(_FileRenameInfoHeader)
        ).contents
        header.Flags = 1 if replace else 0
        header.RootDirectory = ctypes.c_void_p(parent)
        header.FileNameLength = len(encoded)
        ctypes.memmove(
            ctypes.addressof(buffer) + _FileRenameInfoHeader.FileName.offset,
            encoded,
            len(encoded),
        )
        io_status = _IoStatusBlock()
        status = self.NtSetInformationFile(
            handle,
            ctypes.byref(io_status),
            buffer,
            total,
            _FILE_RENAME_INFORMATION_CLASS,
        )
        if status < 0:
            error = int(self.RtlNtStatusToDosError(status))
            if error in {_ERROR_ALREADY_EXISTS, _ERROR_FILE_EXISTS}:
                raise FileExistsError(error, name)
            raise OSError(error, f"NtSetInformationFile(FileRenameInformation) failed for {name!r}")

    def delete(self, handle: int) -> None:
        disposition = _FileDispositionInfo(DeleteFile=1)
        if not self.SetFileInformationByHandle(
            handle,
            _FILE_DISPOSITION_INFO_CLASS,
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            self._raise_last_error(
                "SetFileInformationByHandle(FileDispositionInfo)"
            )

    def close(self, handle: int) -> None:
        if not self.CloseHandle(handle):
            self._raise_last_error("CloseHandle")

    @contextmanager
    def lock(self, handle: int) -> Iterator[None]:
        overlapped = _Overlapped()
        if not self.LockFileEx(
            handle,
            _LOCKFILE_EXCLUSIVE_LOCK | _LOCKFILE_FAIL_IMMEDIATELY,
            0,
            1,
            0,
            ctypes.byref(overlapped),
        ):
            get_last_error = getattr(ctypes, "get_last_error", None)
            if get_last_error is None:
                raise WindowsJournalUnavailable(
                    "ctypes Windows last-error support is unavailable"
                )
            error = int(get_last_error())
            if error == _ERROR_LOCK_VIOLATION:
                raise TimeoutError("Windows metadata writer lock is held")
            self._raise_last_error("LockFileEx")
        try:
            yield
        finally:
            if not self.UnlockFileEx(
                handle, 0, 1, 0, ctypes.byref(overlapped)
            ):
                self._raise_last_error("UnlockFileEx")


class _WindowsHeldDirectory:
    """A directory pinned by an NT handle and its ``FILE_ID_INFO`` identity.

    ``parent``/``name`` are how :meth:`reverify` re-resolves this directory.
    Reading ``handle_info`` back from ``self._handle`` returns the identity
    that handle was opened with, so a check written that way can never fail --
    ``_windows_metadata_journal.py:146-156`` has exactly that bug today.
    """

    def __init__(
        self,
        path: Path,
        handle: int,
        api: _IdentityWindowsApi,
        owned: bool,
        *,
        parent: "_WindowsHeldDirectory | None" = None,
        name: str | None = None,
    ) -> None:
        self.path = path
        self._handle = handle
        self._api = api
        self._owned = owned
        self._parent = parent
        self._name = name
        self._closed = False
        info = api.handle_info(handle)
        if info.attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise IdentityRefused(f"reparse point: {path}")
        # The journal already refuses an absent or all-zero file id, seen on
        # some network filesystems. No stable identity fails closed.
        if len(info.file_id) != 16 or not any(info.file_id):
            raise IdentityRefused(f"no stable identity: {path}")
        if not api.is_directory(handle):
            raise IdentityRefused(f"not a directory: {path}")
        self._identity = info.identity
        self._children: list[_WindowsHeldDirectory] = []

    def _entry_opens_as_a_directory(self, name: str) -> bool:
        """Whether *name* in this held parent opens as a directory.

        The probe goes through the held handle and is closed immediately; it
        is only ever compared against, never opened through, so it reopens no
        TOCTOU window.
        """
        try:
            probe = self._api.open_directory(
                self._handle, name, create=False, share_delete=True
            )
        except OSError:
            return False
        self._api.close(probe)
        return True

    def _refuse_type_mismatch(self, exc: OSError, name: str) -> None:
        """Raise :class:`IdentityRefused` if *exc* is refusal 2 or 3.

        ``_nt_open`` maps only "no such file" to :class:`FileNotFoundError`;
        everything else arrives as a bare :class:`OSError` carrying a *Win32*
        error code, so a type mismatch would otherwise surface as a transport
        failure and the contract tests expecting a refusal would fail. The
        translation lives here rather than inside ``_nt_open`` so the
        journal's error vocabulary is untouched.

        ``ERROR_DIRECTORY`` is unambiguous. ``ERROR_ACCESS_DENIED`` is not:
        ``RtlNtStatusToDosError`` folds ``STATUS_FILE_IS_A_DIRECTORY`` onto
        the code a genuine ACL denial also produces. Rather than guess, re-ask
        the held parent for the same name as a directory -- it succeeds only
        when the entry really is one (or a junction, which
        ``FILE_OPEN_REPARSE_POINT`` opens as itself), and a real permission
        failure is left to propagate as the :class:`OSError` it is.

        Returns without raising when *exc* is neither, so the caller re-raises.
        """
        if exc.errno == _ERROR_DIRECTORY:
            raise IdentityRefused(
                f"not a directory: {self.path / name}"
            ) from exc
        if exc.errno != _ERROR_ACCESS_DENIED:
            return
        if self._entry_opens_as_a_directory(name):
            raise IdentityRefused(
                f"not a regular file: {self.path / name}"
            ) from exc

    def child_directory(self, name: str) -> "HeldDirectory":
        validate_component(name)
        try:
            handle = self._api.open_directory(
                self._handle, name, create=False, share_delete=True
            )
        except OSError as exc:
            self._refuse_type_mismatch(exc, name)
            raise
        try:
            child = _WindowsHeldDirectory(
                self.path / name,
                handle,
                self._api,
                owned=True,
                parent=self,
                name=name,
            )
        except BaseException:
            self._api.close(handle)
            raise
        self._children.append(child)
        return child

    def list_names(self) -> tuple[str, ...]:
        """Return this directory's entries, excluding ``.`` and ``..``.

        Windows reports both; POSIX's ``os.listdir`` reports neither. The
        filter lives here so it is one implementation across every API this
        class runs against, and so a lane that never reaches Win32 still
        exercises it.
        """
        self.reverify()
        return tuple(
            sorted(
                name
                for name in self._api.list_names(self._handle)
                if name not in {".", ".."}
            )
        )

    def _open_regular(self, name: str) -> int:
        validate_component(name)
        try:
            handle = self._api.open_regular_read(self._handle, name)
        except OSError as exc:
            self._refuse_type_mismatch(exc, name)
            raise
        try:
            info = self._api.handle_info(handle)
            # ``FILE_OPEN_REPARSE_POINT`` opens the link itself rather than
            # its target, so the refusal is this check, not the open.
            if info.attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
                raise IdentityRefused(f"reparse point: {self.path / name}")
            # Belt-and-braces against ``FILE_NON_DIRECTORY_FILE``, mirroring
            # the POSIX backend's explicit ``S_ISREG`` check.
            if self._api.is_directory(handle):
                raise IdentityRefused(f"not a regular file: {self.path / name}")
            if self._api.link_count(handle) != 1:
                raise IdentityRefused(
                    f"not a single-link file: {self.path / name}"
                )
        except BaseException:
            self._api.close(handle)
            raise
        return handle

    def read_regular_bytes(
        self, name: str, *, max_bytes: int | None = None
    ) -> bytes:
        payload, _stat = self.read_regular_with_stat(name, max_bytes=max_bytes)
        return payload

    def read_regular_with_stat(
        self, name: str, *, max_bytes: int | None = None
    ) -> tuple[bytes, os.stat_result]:
        """Bytes plus the stat CPython would build for the same file.

        The stat comes from :func:`os.fstat` on a descriptor adopted from the
        held handle, so it is the *same* :class:`os.stat_result` shape the path
        branch of ``store_publication_token`` produces -- the two token
        branches then agree by construction rather than by hope.
        """
        handle = self._open_regular(name)
        # Ownership of the handle transfers to the descriptor here; closing
        # both would be a double close. Until the adoption returns, the handle
        # is still ours, so a failing adoption has to close it.
        try:
            descriptor = self._api.adopt_descriptor(handle)
        except BaseException:
            self._api.close(handle)
            raise
        try:
            before = os.fstat(descriptor)
            if max_bytes is not None and before.st_size > max_bytes:
                raise IdentityRefused(
                    f"{self.path / name} is larger than {max_bytes} bytes"
                )
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                payload = stream.read()
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        changed = (
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_ino,
        ) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_ino,
        )
        if changed or len(payload) != after.st_size:
            raise IdentityRefused(f"{self.path / name} changed during the read")
        self.reverify()
        return payload, after

    def open_regular_stream(self, name: str) -> BinaryIO:
        handle = self._open_regular(name)
        try:
            return self._api.stream(handle)
        except BaseException:
            self._api.close(handle)
            raise

    def reverify(self) -> None:
        """Re-resolve this directory's NAME and compare it to what is held.

        A child re-resolves one component inside its held parent, so no
        ancestor is re-walked. The root has no held parent and re-opens its own
        absolute path -- it catches its own rename, deletion or replacement,
        but not the swap of an ancestor. That asymmetry is deliberate and is
        why the spec's refusal 5 binds children.
        """
        if self._parent is None or self._name is None:
            probe = self._api.open_anchor(
                _extended_length(self.path), share_delete=True
            )
        else:
            probe = self._api.open_directory(
                self._parent._handle,
                self._name,
                create=False,
                share_delete=True,
            )
        try:
            if self._api.handle_info(probe).identity != self._identity:
                raise IdentityRefused(f"identity changed: {self.path}")
        finally:
            self._api.close(probe)

    def _close(self) -> None:
        """Close children leaf-first, then self. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for child in self._children:
            child._close()
        self._api.close(self._handle)


# Bind at import, not per call. ``_CtypesWindowsApi()`` raises
# ``WindowsJournalUnavailable`` from ``__init__`` when a symbol fails to bind.
# Constructing it inside ``open_identity_directory`` would leave
# ``identity_io_available()`` True and ``active_backend_name()`` "windows"
# while every call raised -- the per-platform backend assertion would pass
# while nothing worked, and consumers would see an uncaught RuntimeError
# instead of the designed refusal.
try:
    _API: _CtypesWindowsApi | None = (
        _CtypesWindowsApi() if os.name == "nt" else None
    )
except WindowsJournalUnavailable:  # pragma: no cover - Windows-only path
    _API = None

SUPPORTED = _API is not None


def _extended_length(path: Path) -> str:
    r"""Return the ``\\?\`` spelling of *path* WITHOUT resolving links.

    Not ``ngff_.long_path``: that calls ``Path.resolve()``
    (``ngff_.py:1659``), which follows a junction. A store root that *is* a
    junction would then open successfully here while POSIX's ``O_NOFOLLOW``
    refuses it -- the two backends would stop refusing identically, which is
    the one property the spec insists on.

    Only the initial absolute open needs this: the walk below the root takes
    one component at a time relative to a held handle, where ``MAX_PATH``
    never applies.
    """
    text = os.path.abspath(os.fspath(path))
    return text if text.startswith("\\\\?\\") else "\\\\?\\" + text


@contextmanager
def open_identity_directory(
    path: Path, *, api: _IdentityWindowsApi | None = None
) -> Iterator["HeldDirectory"]:
    """Hold *path* by identity through NT handles.

    Args:
        path: Absolute directory path to hold.
        api: Handle API to drive. Production passes nothing and reuses the
            module-level binding; tests inject an in-memory model.
    """
    root = Path(os.path.abspath(os.fspath(path)))
    # Reuse the module-level instance; it holds only bound function pointers
    # and no per-call state. The journal constructs its own, so a Windows
    # process holds two -- wasteful, not a problem, and worth saying so
    # because a reader will wonder.
    resolved_api: _IdentityWindowsApi | None = api if api is not None else _API
    if resolved_api is None:
        raise IdentityIoUnavailable(
            "the Windows identity-IO backend is unavailable on this platform"
        )
    held: _WindowsHeldDirectory | None = None
    handle = resolved_api.open_anchor(_extended_length(root), share_delete=True)
    try:
        held = _WindowsHeldDirectory(root, handle, resolved_api, owned=True)
        yield held
    finally:
        # In the ``finally``, not after the ``yield``: when the ``with`` body
        # raises, the generator resumes *at* the yield and anything below it is
        # skipped, leaking every child handle. In the Browse route that is one
        # leak per failing request -- the path most exercised under load.
        if held is not None:
            held._close()
        else:
            resolved_api.close(handle)


# Deliberately at the BOTTOM, and the placement is load-bearing.
# ``_identity_io`` runs ``_select_backend()`` at *its* bottom, which reads
# ``SUPPORTED`` off this module. ``_windows_metadata_journal`` imports this
# module at its top, so on Windows a journal-first import would reach
# ``_select_backend()`` while this module was still executing its own header --
# and ``SUPPORTED`` would not exist yet. Importing the facade here, after
# everything this module defines, makes both orders work: a journal-first
# import finds ``SUPPORTED`` bound, and a facade-first import finds these four
# names bound, because all of them are defined above ``_select_backend()``'s
# call site in ``_identity_io``.
from ._identity_io import (  # noqa: E402
    HeldDirectory,
    IdentityIoUnavailable,
    IdentityRefused,
    validate_component,
)
