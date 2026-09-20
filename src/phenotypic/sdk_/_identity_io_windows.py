"""Handle-bound Windows primitives for identity-bound directory I/O.

This module owns only the native Win32/NT plumbing: every child is opened
relative to a held directory handle via ``NtCreateFile``, identity is read
from ``FILE_ID_INFO``, and reparse points are visible in handle metadata
rather than followed. It carries no journal, receipt or transaction
semantics -- :mod:`phenotypic.sdk_._windows_metadata_journal` imports these
names and layers its own semantics on top.

:class:`WindowsJournalUnavailable` lives here rather than with the journal
because :class:`_CtypesWindowsApi` raises it, and a module that raises an
exception defined by its own importer is a cycle.
"""

from __future__ import annotations

import ctypes
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Protocol


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
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_OPEN_EXISTING = 3
_FILE_ATTRIBUTE_TAG_INFO_CLASS = 9
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
