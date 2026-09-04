"""The in-process verification cache.

**Stub.** The body -- ``CachedVerification``, ``cached_states``,
``remember_states``, ``entry_is_still_current``, ``clear_verification_cache``
and ``tracked_output_count`` -- lands in Phase 1 Task 3, together with the
INV-VERDICT mutation suite that is its whole correctness argument. This file
exists now so that the module boundary, and the import test that pins it, are
in place before any of that is written.

INVARIANT (INV-VERDICT) -- **nothing may improve a verdict except a successful
deep verification.** No function here will return a verdict to a caller that
has not deep-verified; an entry only licenses a *previously* deep-verified
result standing while its stat tuples are unchanged. A stale, replaced or
forged entry therefore degrades to today's behaviour and never past it.

Decision D-B keeps this in process rather than in
``.phenotypic/verification_cache.json``: audit S1 proposed a process-level
cache and spec §9.1 escalated it to a file without cause. Every cadence the
audit measured is a repeated call inside one long-lived process.

Imports :class:`~phenotypic.sdk_._state_types.ImageState` from the leaf module,
never from :mod:`phenotypic.sdk_._run_state` -- that direction is the cycle the
three-module split exists to prevent. Nothing here may import
:mod:`phenotypic._cli` (INV-LAYER).
"""

from __future__ import annotations
