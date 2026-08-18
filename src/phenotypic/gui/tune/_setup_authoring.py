"""Back-compat shim. Implementation in :mod:`phenotypic._services.tune_spec`.

Re-exports the *same* objects, so a ``SetupDraft`` written by the Setup view and
one written by the MCP server are the same class. ``_callbacks.py:72`` and two
test modules import through here.

The surface below is every name the repo imports from this module, derived by
parsing the imports rather than from the old ``__all__`` — ``write_setup_draft``
and ``build_authored_setup_spec`` are pulled by
``tests/integration/gui/tune/test_setup_view.py`` through multi-line
parenthesised imports that a single-line grep does not see.
"""

from __future__ import annotations

from phenotypic._services.tune_spec import (  # noqa: F401
    SETUP_DRAFT_VERSION,
    SetupAuthoringResult,
    SetupDraft,
    SetupDraftCache,
    SetupPathPayload,
    SetupPathResolution,
    SetupWriteReceipt,
    authored_content_fingerprint,
    authored_setup_spec_path,
    build_authored_setup_spec,
    build_setup_draft,
    load_pipeline_or_spec,
    path_content_fingerprint,
    resolve_picker_payload,
    resolve_setup_path,
    setup_draft_from_store,
    setup_path_payload,
    setup_path_resolution_from_store,
    write_authored_setup_spec,
    write_setup_draft,
    write_setup_draft_receipt,
)

__all__ = [
    "SETUP_DRAFT_VERSION",
    "SetupAuthoringResult",
    "SetupDraft",
    "SetupDraftCache",
    "SetupPathPayload",
    "SetupPathResolution",
    "SetupWriteReceipt",
    "authored_content_fingerprint",
    "authored_setup_spec_path",
    "build_authored_setup_spec",
    "build_setup_draft",
    "load_pipeline_or_spec",
    "path_content_fingerprint",
    "resolve_picker_payload",
    "resolve_setup_path",
    "setup_draft_from_store",
    "setup_path_payload",
    "setup_path_resolution_from_store",
    "write_authored_setup_spec",
    "write_setup_draft",
    "write_setup_draft_receipt",
]
