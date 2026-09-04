"""Frozen state types shared by the run-state readers.

**The leaf of a three-module chain.** Dependency order is strictly
``_state_types`` ← :mod:`phenotypic.sdk_._verification_cache` ←
:mod:`phenotypic.sdk_._run_state`, with no edge back. That shape exists to
break a real cycle: the cache stores whole :class:`ImageState` objects, so it
needs the type, while ``_run_state`` needs the cache. Hoisting the dataclasses
into a module with no behaviour of its own resolves it and costs nothing.

**Data only.** No logic, no I/O, and -- like both modules above it -- nothing
from :mod:`phenotypic._cli`. INV-LAYER
(``tests/unit/sdk_/test_run_state_layering.py``) binds all three.

The four classes are re-exported from :mod:`phenotypic.sdk_._run_state`, which
is where spec §5.2 declares the public surface. Import them from there, or from
:mod:`phenotypic.sdk_`; this module is an implementation detail of the split.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Completion = Literal["complete", "incomplete", "failed", "active"]
Depth = Literal["shallow", "deep"]

#: One image's verdict. Named rather than spelled inline on
#: :class:`ImageState` so that :mod:`phenotypic.sdk_._run_state` can annotate
#: the value it computes with the same type the field declares, instead of
#: repeating the three strings in a second place -- which is the duplication
#: this whole change is about, at its smallest scale.
Verdict = Literal["verified", "unverified", "failed"]


@dataclass(frozen=True)
class RunIdentity:
    """The run-level identity of one run configuration (spec §5.1, §5.3).

    Carries four of §5.1's five identity tokens -- the fifth, ``work_id``, is
    per-image and lives on :class:`ImageState` -- plus §5.3's three digests.

    ``processing_generation`` is content-derived (from P2 onward), so resume and
    fencing are emergent rather than bookkeeping: two invocations with the same
    inputs mint the same identity without either having read the other's state.

    Note the count: spec §5.1 is headed "the six tokens" and its own amendment
    U-4 cuts ``publication_id``, leaving five, of which ``work_id`` and
    ``processing_generation`` are the content-derived pair. Anything still
    saying "six tokens, three content-derived" predates U-4.
    """

    processing_generation: str
    restart_epoch: int
    scheduler_epoch: str | None
    owner_generation: str | None
    inventory_digest: str
    scientific_config_digest: str
    finalization_input_digest: str

    def digest(self) -> str:
        """Return a stable digest of the fencing-relevant tokens.

        ``scheduler_epoch`` and ``owner_generation`` are excluded: they are
        liveness facts, not configuration, and folding them in would discard
        the verification cache every time a job is submitted against unchanged
        work.
        """
        payload = {
            "processing_generation": self.processing_generation,
            "restart_epoch": self.restart_epoch,
            "inventory_digest": self.inventory_digest,
            "scientific_config_digest": self.scientific_config_digest,
            "finalization_input_digest": self.finalization_input_digest,
        }
        return hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class ImageState:
    """One image's stages and whether its declared artifacts still match disk.

    ``stages`` is the open map from spec §6.1 -- ``stage1``/``stage2``/
    ``stage3``/``measured`` today, more later. Nothing here enumerates its
    keys; a caller asking "did stage 3 run?" reads ``"stage3" in state.stages``,
    which is what makes a future stage additive rather than a schema break.

    Under D-A there is no ``backfilled`` stage: per-store metadata is written in
    the store's original promote, so there is nothing to record having happened
    afterwards.
    """

    work_id: str
    dataset: str
    image_stem: str
    stages: Mapping[str, Mapping[str, object]]
    #: Spec §9 annotates ``images`` as "work_id -> stages + VERDICT". A bool
    #: plus an unread ``reason`` was not that (SIMP-R1-09).
    verdict: Verdict
    reason: str | None = None


@dataclass(frozen=True)
class RunDiagnostics:
    """Counts derived from ``images``. **Nothing branches on these** (§4.2, §9).

    One-line projections over :attr:`ImageState.verdict`, not cached counts of a
    collection the caller already holds.

    ``manifest.json``'s counts and the event log's presence were in an earlier
    draft of this dataclass and are **dropped** (U-5): verified zero consumers
    survive P6, and carrying demoted evidence into :class:`RunState` is what
    keeps it alive as a quasi-evidence surface. The files remain on disk for a
    human debugging a run.
    """

    accepted: int
    verified: int
    failed: int


@dataclass(frozen=True)
class RunState:
    """The single answer to "is this run done?" (spec §4.3, §9)."""

    completion: Completion
    identity: RunIdentity
    images: Mapping[str, ImageState]
    advisories: tuple[str, ...]
    diagnostics: RunDiagnostics
    depth: Depth
    verified_at: datetime | None = None
