from __future__ import annotations

# Schema-aware metadata prefixing lives in ``phenotypic.sdk_`` so the post
# package does not pull in the core image stack. Re-exported here for the post
# ops (and any legacy callers) that import it from this module.
from phenotypic.sdk_ import ensure_metadata_prefix

__all__ = ["ensure_metadata_prefix"]
