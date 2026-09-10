"""One canonical JSON digest, for every producer and every reader.

**Hoisted, not copied (CAN-29).** The function below existed twice --
``_cli_completion._canonical_digest`` and
``_cli_failure_tracker._canonical_digest``. :mod:`phenotypic.sdk_._run_state`
needs it as well, and INV-LAYER forbids that module importing either CLI
home, so the choice was between a third copy pinned against the other two by
a keeper test and one shared definition. This is the shared definition: it
removes two copies rather than adding one, and it leaves nothing that could
disagree.

**The two copies were not byte-identical, and the difference was checked
rather than assumed.** They differed in exactly three things: the parameter
name (``value`` vs ``payload``), its annotation (``object`` vs
``Mapping[str, Any]``), and line wrapping. The ``json.dumps`` call is
argument-for-argument the same, which is a proof over *all* inputs -- a
single probe agreeing on one dict would not have been, since a digest
function's whole job is to disagree when inputs differ.

The hoist takes the **wider** annotation, ``object``, and that direction is
deliberate: every call site satisfying ``Mapping[str, Any]`` also satisfies
``object``, so no existing caller can newly fail to type-check. Hoisting the
narrower one would have left ``_cli_completion``'s sites -- which pass
lists -- rejected.

Hoisting a **pure function with no I/O** does not breach P1's "moves no
consumers" rule. That rule is about *state* consumers -- the reason this
phase's correctness can be established in isolation -- and nothing here can
change a verdict.

``ensure_ascii=False`` is load-bearing (ledger DF-19), not a style choice.
Both original copies used it, so every proof already on disk was written that
way; flipping it would make ``canonical_digest`` disagree with itself on any
non-ASCII dataset name and invalidate every proof written by the other half
of the code.
"""

from __future__ import annotations

import hashlib
import json

__all__ = ["canonical_digest"]


def canonical_digest(value: object) -> str:
    """Return the SHA-256 of ``value``'s canonical JSON encoding.

    Canonical means sorted keys, no insignificant whitespace, and literal
    non-ASCII characters. Two structurally equal values therefore digest
    identically regardless of how either was built.

    Args:
        value: Any JSON-serializable value. Mappings are the usual case --
            an inventory, a finalization-input object, a sorted work-id list.

    Returns:
        The 64-character lowercase hex digest.

    Raises:
        TypeError: If ``value`` is not JSON-serializable.

    Example:
        >>> from phenotypic.sdk_._digests import canonical_digest
        >>> canonical_digest({"b": 1, "a": 2}) == canonical_digest(
        ...     {"a": 2, "b": 1}
        ... )
        True
    """
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
