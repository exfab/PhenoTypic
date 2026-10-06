"""Capability mixin: an operation that reads per-image reference metadata."""

from __future__ import annotations

import functools
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from phenotypic._core._image import Image

__all__ = ["RefMetadata"]

#: What each op resolved during its current apply, keyed by ``id(op)``, read
#: once by ``provenance_parameters`` right after the op succeeds.
_RESOLVED: ContextVar[dict[int, dict[str, Any]] | None] = ContextVar(
    "phenotypic_ref_metadata_resolved", default=None
)


def _resolved() -> dict[int, dict[str, Any]]:
    store = _RESOLVED.get()
    if store is None:
        store = {}
        _RESOLVED.set(store)
    return store


def _discard_record_on_failure(operate: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a subclass's ``_operate`` so a failed apply drops its record.

    ``provenance_parameters`` pops the record only after success, so without
    this a failure leaves it keyed by ``id(op)``; a later op given the same id
    that skips its lookup would inherit it. ``apply`` cannot be overridden for
    this, because RefMetadata may sit last in the MRO.
    """

    @functools.wraps(operate)
    def _operate(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return operate(self, *args, **kwargs)
        except BaseException:
            _resolved().pop(id(self), None)
            raise

    return _operate


def _marked_fields(cls: type, marker_type: type, *, source: str | None = None) -> tuple[str, ...]:
    names: list[str] = []
    for name, info in cls.model_fields.items():  # type: ignore[attr-defined]
        for marker in info.metadata:
            if isinstance(marker, marker_type) and (
                source is None or getattr(marker, "source", None) == source
            ):
                names.append(name)
                break
    return tuple(names)


class RefMetadata:
    """Mark an operation that reads per-image values from a ReferenceContext.

    Fieldless, like :class:`~phenotypic.abc_.plotting.PlotImage`. A subclass
    declares its columns as ordinary fields typed
    :data:`~phenotypic.sdk_.RefColumn` (a value) or
    :data:`~phenotypic.sdk_.RefImageColumn` (a value naming another image),
    then calls :meth:`_ref_values` and :meth:`_ref_image` inside ``_operate``.

    The operation never holds a table path: the table is supplied when the
    pipeline runs — ``with phenotypic.ReferenceContext(...)`` in Python, the
    CLI's ``--metadata``, or the GUI's reference-metadata picker — so using
    metadata is always a deliberate act at run time.

    Raises:
        TypeError: At class definition, when the subclass is not an
            ``ImageOperation``.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        from phenotypic.abc_._image_operation import ImageOperation

        if not issubclass(cls, ImageOperation):
            raise TypeError(
                f"{cls.__name__}: RefMetadata may only be mixed into an ImageOperation"
            )
        operate = cls.__dict__.get("_operate")
        if operate is not None and not getattr(operate, "__isabstractmethod__", False):
            cls._operate = _discard_record_on_failure(operate)  # type: ignore[method-assign]

    def _ref_columns(self) -> tuple[str, ...]:
        """Every column this op reads, in field declaration order."""
        from phenotypic.sdk_._column_ref import _ColumnRefMarker

        fields = _marked_fields(type(self), _ColumnRefMarker, source="reference_metadata")
        return tuple(getattr(self, name) for name in fields)

    def _ref_image_columns(self) -> tuple[str, ...]:
        """The subset of :meth:`_ref_columns` whose values name images."""
        from phenotypic.sdk_._column_ref import _ReferenceImageMarker

        fields = _marked_fields(type(self), _ReferenceImageMarker)
        return tuple(getattr(self, name) for name in fields)

    def _require_context(self):
        from phenotypic._core._reference_context import (
            ReferenceContext,
            RefMetadataUnavailableError,
        )

        ctx = ReferenceContext.current()
        if ctx is None:
            raise RefMetadataUnavailableError(
                f"{type(self).__name__} reads {self._ref_columns()} from a "
                f"ReferenceContext, but none is active.\n"
                f"  Python: with phenotypic.ReferenceContext('layout.csv', "
                f"image_root='images/'): pipe.apply(img)\n"
                f"  CLI:    pass --metadata layout.csv"
            )
        return ctx

    def _ref_values(self, image: "Image") -> dict[str, str]:
        """Look up this op's columns for *image* in the active context."""
        ctx = self._require_context()
        values = ctx.lookup(image, self._ref_columns())
        _resolved()[id(self)] = {
            "table_sha256": ctx.table_sha256,
            "values": dict(values),
            "images": {},
        }
        return values

    def _ref_image(self, name: str) -> "Image":
        """Load the reference image *name* through the active context.

        Call :meth:`_ref_values` first in the same ``_operate``: it resets this
        op's provenance record, so a record left by an earlier apply that
        raised cannot leak into this one.
        """
        ctx = self._require_context()
        # One load for both: the digest recorded is that of the pixels used.
        image, digest = ctx._load(name)
        record = _resolved().setdefault(
            id(self), {"table_sha256": ctx.table_sha256, "values": {}, "images": {}}
        )
        record["images"][name] = {"sha256": digest}
        return image

    def provenance_parameters(self) -> dict[str, Any]:
        """Parameters for the provenance journal, plus what this apply resolved."""
        params = self.model_dump(mode="json")  # type: ignore[attr-defined]
        record = _resolved().pop(id(self), None)
        if record is not None:
            params["_references"] = record
        return params
