"""Central authorization for persistent Results and Analysis mutations.

Read-only output discovery intentionally accepts incomplete and contradictory
run evidence so users can still inspect whatever artifacts are present. That
is separate from write authority. Every GUI mutation must obtain a fresh
receipt from :class:`OutputMutationGuard` immediately before writing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final

from flask import current_app, has_app_context, has_request_context, request

from phenotypic.gui._binding_generation import (
    BINDING_GENERATION_PAYLOAD_KEY,
)
from phenotypic.gui._config import CFG_OUTPUT_MUTATION_GUARD
from phenotypic.gui.results_viewer._output_consistency import (
    inspect_output_consistency,
)

if TYPE_CHECKING:
    from phenotypic.gui.results_viewer._output_root import OutputRoot

_PRESENTED_FROM_REQUEST: Final = object()


class OutputMutationBlocked(RuntimeError):
    """Raised before a write when the bound output is not authoritative."""


@dataclass(frozen=True)
class OutputMutationReceipt:
    """Fresh evidence authorizing one mutation attempt."""

    action: str
    binding_generation: str | None
    processing_fingerprint: str
    consistency_evidence_fingerprint: str
    authorized_at: datetime


@dataclass(frozen=True)
class OutputMutationGuard:
    """Bind one output snapshot and browser generation to every GUI write."""

    output_root: OutputRoot
    binding_generation: str | None

    def authorize(
        self,
        action: str,
        *,
        presented_generation: str | None | object = _PRESENTED_FROM_REQUEST,
    ) -> OutputMutationReceipt:
        """Return a fresh receipt or fail closed before the caller writes.

        Args:
            action: Reader-facing mutation name used in diagnostics.
            presented_generation: Browser generation to compare with this
                bound app. The default reads Dash's request payload. Tests and
                non-request callers may pass an explicit value.

        Returns:
            An immutable receipt recording the evidence checked immediately
            before the mutation.

        Raises:
            OutputMutationBlocked: If generation, consistency, ownership, or
                the processing snapshot is stale.
        """
        supplied = (
            _presented_request_generation(self.binding_generation)
            if presented_generation is _PRESENTED_FROM_REQUEST
            else presented_generation
        )
        if supplied != self.binding_generation:
            raise OutputMutationBlocked(
                f"{action} blocked: this page belongs to an older output "
                "binding. Reload before retrying."
            )

        # A binding discovered from incomplete or contradictory evidence never
        # gains write authority in place. It carries only bounded structural
        # processing assurance and must be refreshed after completion evidence
        # becomes coherent.
        if self.output_root.consistency.is_read_only:
            detail = "; ".join(self.output_root.consistency.reasons)
            raise OutputMutationBlocked(
                f"{action} blocked: output completion evidence is "
                f"{self.output_root.consistency.state}. {detail}"
            )
        if not self.output_root.has_exhaustive_processing_inventory:
            raise OutputMutationBlocked(
                f"{action} blocked: this read-only binding does not carry an "
                "exhaustive processing inventory. Refresh Results and Analysis."
            )

        # Check completion evidence on both sides of the exhaustive inventory
        # verification. This closes the mutation receipt over owner/manifest
        # changes without making read-only bindings walk unrelated artifacts.
        fresh_consistency = inspect_output_consistency(self.output_root.layout)
        if fresh_consistency.state != "coherent":
            detail = "; ".join(fresh_consistency.reasons)
            raise OutputMutationBlocked(
                f"{action} blocked: output completion evidence is "
                f"{fresh_consistency.state}. {detail}"
            )
        if (
            fresh_consistency.evidence_fingerprint
            != self.output_root.consistency.evidence_fingerprint
        ):
            raise OutputMutationBlocked(
                f"{action} blocked: completion evidence changed after this "
                "snapshot was bound. Refresh Results and Analysis."
            )
        if self.output_root.active_run_is_currently_running():
            raise OutputMutationBlocked(
                f"{action} blocked: a nonterminal output owner is active."
            )
        if not self.output_root.snapshot_is_current():
            raise OutputMutationBlocked(
                f"{action} blocked: processing artifacts changed after this "
                "snapshot was bound. Refresh Results and Analysis."
            )
        verified_consistency = inspect_output_consistency(
            self.output_root.layout
        )
        if (
            verified_consistency.state != "coherent"
            or verified_consistency.evidence_fingerprint
            != fresh_consistency.evidence_fingerprint
            or verified_consistency.has_active_owner
        ):
            raise OutputMutationBlocked(
                f"{action} blocked: completion evidence changed while "
                "processing artifacts were verified. Refresh Results and "
                "Analysis."
            )
        return OutputMutationReceipt(
            action=action,
            binding_generation=self.binding_generation,
            processing_fingerprint=(
                self.output_root.snapshot.processing_fingerprint
            ),
            consistency_evidence_fingerprint=(
                fresh_consistency.evidence_fingerprint
            ),
            authorized_at=datetime.now(timezone.utc),
        )


def require_output_mutation(
    action: str,
    *,
    output_root: OutputRoot | None = None,
) -> OutputMutationReceipt:
    """Authorize one mutation through the app guard or an explicit snapshot.

    ``output_root`` is only a fallback for direct, non-request helper calls.
    Browser callbacks always use the guard installed on the current Flask
    application, which also carries the renderer binding generation.
    """
    guard = (
        current_app.config.get(CFG_OUTPUT_MUTATION_GUARD)
        if has_app_context()
        else None
    )
    if guard is None and output_root is not None:
        guard = OutputMutationGuard(output_root, None)
    if not isinstance(guard, OutputMutationGuard):
        raise OutputMutationBlocked(
            f"{action} blocked: no current output mutation authority exists."
        )
    return guard.authorize(action)


def output_mutations_disabled(output_root: OutputRoot) -> bool:
    """Return whether persistent controls must render disabled."""
    return output_root.consistency.is_read_only or (
        output_root.snapshot.active_run
    )


def output_read_only_diagnostic(output_root: OutputRoot) -> str | None:
    """Return a visible diagnostic for a browsable but non-mutable output."""
    if not output_mutations_disabled(output_root):
        return None
    reasons = "; ".join(output_root.consistency.reasons)
    return (
        f"Read-only output: completion evidence is "
        f"{output_root.consistency.state}. {reasons} Persistent QC, "
        "curation, Error, and Analysis actions are disabled. Browsing remains "
        "available; this viewer will not repair or resume the run."
    )


def _presented_request_generation(
    bound_generation: str | None,
) -> str | None:
    """Read Dash's renderer-injected generation from the current request."""
    if not has_request_context():
        # Direct internal calls are already bound to this guard instance.
        # Actual browser requests always take the strict payload branch below.
        return bound_generation
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return None
    value = payload.get(BINDING_GENERATION_PAYLOAD_KEY)
    return value if isinstance(value, str) else None


__all__ = [
    "OutputMutationBlocked",
    "OutputMutationGuard",
    "OutputMutationReceipt",
    "output_mutations_disabled",
    "output_read_only_diagnostic",
    "require_output_mutation",
]
