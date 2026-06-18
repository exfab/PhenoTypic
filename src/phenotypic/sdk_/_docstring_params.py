"""Shared docstring-parameter parsing for the pydantic operation system.

Operation parameter descriptions live in hand-written Google-style
``Args:`` docstring blocks. After the pydantic v2 migration, each
operation parameter becomes an annotated pydantic field, and pydantic's
``model_json_schema()`` exposes a per-field ``description`` slot — the
machine-readable contract an MCP server needs.

This module bridges the two: :func:`parse_param_descriptions` extracts a
``name -> description`` map from a class docstring, and
:func:`apply_docstring_descriptions` copies those descriptions onto a
pydantic model's fields so they surface in the JSON schema.

The parser is ported verbatim from ``LazyWidgetMixin._parse_docstring``
so the existing widget-help behaviour is preserved when that mixin is
later switched over to this shared function.
"""
from __future__ import annotations

import re
from typing import Any

# Section headers introducing a parameter block.
# Google: "Args:", "Arguments:", "Attributes:", "Parameters:".
_SECTION_HEADER_RE = re.compile(
    r"^\s*(Args|Arguments|Attributes|Parameters)\s*:\s*$",
    re.IGNORECASE,
)
# NumPy: a bare "Parameters" line followed by a dashed underline.
_NUMPY_HEADER_RE = re.compile(r"^\s*Parameters\s*$", re.IGNORECASE)
_NUMPY_UNDERLINE_RE = re.compile(r"^\s*-+\s*$")

# Google: "name (type): description" or "name: description".
_GOOGLE_PARAM_RE = re.compile(r"^\s*(\w+)\s*(\(.*?\))?\s*:\s*(.+)$")
# NumPy: "name : type" (description begins on the following lines).
_NUMPY_PARAM_RE = re.compile(r"^\s*(\w+)\s*:\s*(.*)$")
# Sphinx/ReST: ":param name: description" — may appear anywhere.
_SPHINX_PARAM_RE = re.compile(r"^\s*:param\s+(\w+)\s*:\s*(.+)$")


def parse_param_descriptions(doc: str | None) -> dict[str, str]:
    """Extract parameter descriptions from a class docstring.

    Supports Google (``Args:``/``Arguments:``/``Attributes:``/
    ``Parameters:``), NumPy (``Parameters`` + dashed underline), and
    Sphinx/ReST (``:param name:``) styles. Continuation lines are joined
    with single spaces.

    Args:
        doc: A class docstring, or ``None``. ``None`` and the empty
            string both yield an empty mapping.

    Returns:
        dict[str, str]: Mapping of parameter name to its description.
        Parameters with no description block are absent from the map.

    Example:
        >>> doc = '''Detect colonies.
        ...
        ... Args:
        ...     ignore_zeros: Exclude zero-intensity pixels.
        ...     ignore_borders: Drop colonies touching the edge.
        ... '''
        >>> parse_param_descriptions(doc)["ignore_zeros"]
        'Exclude zero-intensity pixels.'
        >>> parse_param_descriptions(None)
        {}
    """
    if not doc:
        return {}

    params: dict[str, str] = {}
    lines = doc.split("\n")

    # State-machine variables.
    in_param_section = False
    current_param: str | None = None
    current_desc: list[str] = []

    def _flush() -> None:
        """Commit the pending parameter's accumulated description."""
        nonlocal current_param, current_desc
        if current_param:
            params[current_param] = " ".join(current_desc).strip()
            current_param = None
            current_desc = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()

        # Sphinx style is matched line-by-line; no section required.
        sphinx_match = _SPHINX_PARAM_RE.match(line)
        if sphinx_match:
            _flush()
            current_param = sphinx_match.group(1)
            current_desc = [sphinx_match.group(2)]
            in_param_section = False
            i += 1
            continue

        # Google-style section header.
        if _SECTION_HEADER_RE.match(stripped_line):
            in_param_section = True
            _flush()
            i += 1
            continue

        # NumPy-style header is a bare line followed by an underline.
        if _NUMPY_HEADER_RE.match(stripped_line):
            if i + 1 < len(lines) and _NUMPY_UNDERLINE_RE.match(
                lines[i + 1]
            ):
                in_param_section = True
                i += 2
                continue

        if in_param_section:
            # End-of-section heuristic: an unindented, non-empty line
            # that is not itself a parameter definition.
            if (
                line
                and not line[0].isspace()
                and not _GOOGLE_PARAM_RE.match(line)
                and not _NUMPY_PARAM_RE.match(line)
            ):
                in_param_section = False
                _flush()
                i += 1
                continue

            # New Google-style parameter definition.
            g_match = _GOOGLE_PARAM_RE.match(line)
            if g_match:
                if current_param:
                    params[current_param] = " ".join(
                        current_desc
                    ).strip()
                current_param = g_match.group(1)
                current_desc = [g_match.group(3)]
                i += 1
                continue

            # New NumPy-style parameter definition.
            n_match = _NUMPY_PARAM_RE.match(line)
            if n_match:
                if current_param:
                    params[current_param] = " ".join(
                        current_desc
                    ).strip()
                current_param = n_match.group(1)
                current_desc = []  # NumPy desc starts on the next line.
                i += 1
                continue

            # Continuation of the current parameter's description.
            if current_param and stripped_line:
                current_desc.append(stripped_line)

        i += 1

    # Commit a parameter still pending at end-of-docstring.
    if current_param:
        params[current_param] = " ".join(current_desc).strip()

    return params


def apply_docstring_descriptions(cls: type[Any]) -> None:
    """Populate a pydantic model's empty field descriptions from its docstring.

    For each field on ``cls`` whose ``description`` is unset, the value
    parsed from the class docstring's ``Args:`` block (via
    :func:`parse_param_descriptions`) is assigned. Fields that already
    carry an explicit description are left untouched. The model is then
    rebuilt with ``model_rebuild(force=True)`` so the new descriptions
    reach ``model_json_schema()`` — without the rebuild, edits to
    ``FieldInfo.description`` after class construction do not propagate.

    No-ops cleanly for classes without ``model_fields`` (e.g. abstract
    bases not yet promoted to pydantic models).

    This is intended to be called from
    ``BaseOperation.__pydantic_init_subclass__`` once the operation tree
    is migrated; it is provided here as a standalone function so it can
    be unit-tested in isolation.

    Args:
        cls: A pydantic ``BaseModel`` subclass (or any class — non-model
            classes are silently skipped).

    Example:
        >>> from pydantic import BaseModel
        >>> class Blur(BaseModel):
        ...     '''Blur an image.
        ...
        ...     Args:
        ...         sigma: Standard deviation of the Gaussian kernel.
        ...     '''
        ...     sigma: float = 1.0
        >>> apply_docstring_descriptions(Blur)
        >>> Blur.model_fields["sigma"].description
        'Standard deviation of the Gaussian kernel.'
    """
    model_fields = getattr(cls, "model_fields", None)
    if not model_fields:
        return

    descriptions = parse_param_descriptions(cls.__doc__)
    if not descriptions:
        return

    changed = False
    for name, field in model_fields.items():
        if field.description is None and name in descriptions:
            field.description = descriptions[name]
            changed = True

    if changed:
        rebuild = getattr(cls, "model_rebuild", None)
        if callable(rebuild):
            rebuild(force=True)
