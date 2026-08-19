"""Conformance harness for written NGFF stores."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # `xmlschema` is imported lazily inside the functions below,
    # so the quoted return annotation has no runtime binding to resolve against
    # and ruff reports F821. Guarding the import here keeps the annotation
    # exactly as specified while leaving the import cost where it was.
    import xmlschema


@functools.lru_cache(maxsize=1)
def _ome_xsd() -> "xmlschema.XMLSchema":
    """Load the vendored OME-XML schema.

    NGFF 2.2.3 makes the document a conditional MUST, and JSON-schema
    validation says nothing about it.
    """
    import xmlschema

    path = Path(__file__).resolve().parent / "fixtures" / "ome" / "2016-06" / "ome.xsd"
    if not path.is_file():
        raise AssertionError(
            f"vendored OME schema missing: {path}. A conformance check that "
            "cannot run must fail, never skip."
        )
    return xmlschema.XMLSchema(str(path))


def assert_ome_xml_valid(xml: str) -> None:
    """Validate an OME-XML document against the vendored `ome.xsd`.

    Catches ``XMLSchemaException``, not ``XMLSchemaValidationError``: the
    narrower class does not cover a well-formedness failure, which raises
    ``XMLResourceParseError`` -- and that is the most likely real failure, since
    a control character in an imported EXIF tag breaks well-formedness rather
    than schema conformance (ledger **ALGO-R2B-11**).

    Args:
        xml: The document as a string.

    Raises:
        AssertionError: On any schema violation OR malformed document.
    """
    import xmlschema

    try:
        _ome_xsd().validate(xml)
    except xmlschema.XMLSchemaException as exc:
        # Name the exception type: the widened catch also takes
        # XMLSchemaOSError/XMLResourceOSError, and a disk error reported as
        # "not valid against ome.xsd" would be misread as a conformance bug.
        raise AssertionError(
            f"OME-XML failed validation against ome.xsd "
            f"[{type(exc).__name__}]: {exc}"
        ) from exc
