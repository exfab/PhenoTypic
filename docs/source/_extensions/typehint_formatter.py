"""
Sphinx extension to format type hints consistently.
"""

import re


# Pattern that matches top-level phenotypic subpackages whose import names
# end in a trailing underscore (``phenotypic.tools_``, ``phenotypic.abc_``,
# ``phenotypic.settings_``) when they appear in a fully qualified Python path
# (i.e. followed by a dot). docutils reads the trailing underscore as a
# hyperlink-reference marker (``name_`` -> reference to target ``name``) and
# emits ``WARNING: Unknown target name: "phenotypic.<pkg>"`` for every
# autodoc-generated ``:type:`` field that mentions these subpackages (the
# dataclasses in ``phenotypic.tools_.branch_pathfinding._dataclasses`` are
# the canonical trigger). Escaping the underscore (``tools\_``) keeps the
# rendered output identical while stopping docutils from parsing it as a
# reference.
_UNDERSCORE_PKG_RE = re.compile(r"phenotypic\.(\w+)_(?=\.)")


def setup(app):
    """
    Setup function for the extension.
    """
    # Connect to the autodoc-process-signature event
    app.connect("autodoc-process-signature", process_signature)
    # Connect to autodoc-process-docstring to escape trailing underscores in
    # fully qualified type paths (see _UNDERSCORE_PKG_RE docstring).
    app.connect("autodoc-process-docstring", escape_underscore_package_refs)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }


def process_signature(app, what, name, obj, options, signature, return_annotation):
    """
    Process the signature to clean up type hints.
    """
    # Define replacements for common problematic type hints
    replacements = {
        "matplotlib.axes._axes.Axes": "matplotlib.axes.Axes",
        "<class 'matplotlib.axes._axes.Axes'>": "matplotlib.axes.Axes",
        "matplotlib.figure.Figure": "matplotlib.figure.Figure",
        "<class 'matplotlib.figure.Figure'>": "matplotlib.figure.Figure",
    }

    # Process return annotation
    if return_annotation:
        for old, new in replacements.items():
            if old in return_annotation:
                return_annotation = return_annotation.replace(old, new)

    # Process signature
    if signature:
        for old, new in replacements.items():
            if old in signature:
                signature = signature.replace(old, new)

    return signature, return_annotation


def escape_underscore_package_refs(app, what, name, obj, options, lines):
    """Escape trailing-underscore phenotypic subpackages in type paths.

    Autodoc emits ``:type:`` field bodies as raw text. When the type is
    fully qualified under ``phenotypic.<pkg>_.<subpkg>...`` docutils reads
    the trailing underscore as an anonymous hyperlink reference and fails
    to resolve it, producing a ``WARNING: Unknown target name:
    "phenotypic.<pkg>"`` warning. Rewriting ``<pkg>_.`` to ``<pkg>\\_.``
    keeps the rendered text identical while neutralising the reference
    parse.
    """
    for i, line in enumerate(lines):
        if _UNDERSCORE_PKG_RE.search(line):
            lines[i] = _UNDERSCORE_PKG_RE.sub(r"phenotypic.\1\\_", line)
