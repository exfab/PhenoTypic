"""Central authorization for persistent Results and Analysis mutations.

Read-only output discovery intentionally accepts incomplete, failed and
active runs so users can still inspect whatever artifacts are present. That
is separate from write authority. Every GUI mutation must obtain a fresh
receipt from :class:`OutputMutationGuard` immediately before writing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final

from flask import current_app, has_app_context, has_request_context, request

from phenotypic._gui._binding_generation import (
    BINDING_GENERATION_PAYLOAD_KEY,
)
from phenotypic._gui._config import CFG_OUTPUT_MUTATION_GUARD
from phenotypic._gui.results_viewer._output_root import (
    resolve_output_run_state,
)
from phenotypic.sdk_ import RunState

if TYPE_CHECKING:
    from phenotypic._gui.results_viewer._output_root import OutputRoot

_PRESENTED_FROM_REQUEST: Final = object()


# `resolve_output_run_state` returns None for a standalone deliverables
# bundle. `OutputRoot` centralizes that for the state it holds; these three do
# the same for a state resolved fresh inside `authorize`, so the bundle case
# is decided once per question rather than at each of the six comparisons.


def _run_is_complete(state: RunState | None) -> bool:
    """Whether a freshly resolved state authorizes a write.

    ``True`` for a bundle: it has no run directory, so there is no run to be
    unfinished -- the same answer the retired classifier gave by returning
    ``coherent`` for ``standalone_bundle=True``.
    """
    return state is None or state.completion == "complete"


def _completion(state: RunState | None) -> str:
    """The completion token for a diagnostic, or ``"complete"`` for a bundle."""
    return "complete" if state is None else state.completion


def _advisories(state: RunState | None) -> tuple[str, ...]:
    """A freshly resolved state's advisories, or empty for a bundle."""
    return () if state is None else state.advisories


def _identity_digest(state: RunState | None) -> str:
    """A run's fencing digest, or ``""`` for a bundle."""
    return "" if state is None else state.identity.digest()


class OutputMutationBlocked(RuntimeError):
    """Raised before a write when the bound output is not authoritative."""


@dataclass(frozen=True)
class OutputMutationReceipt:
    """Fresh evidence authorizing one mutation attempt."""

    action: str
    binding_generation: str | None
    processing_fingerprint: str
    run_identity_digest: str
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
            OutputMutationBlocked: If generation, run state, ownership, or
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

        # A binding discovered from an unfinished run never gains write
        # authority in place. It carries only bounded structural processing
        # assurance and must be refreshed after the run completes.
        if not self.output_root.run_is_complete:
            detail = "; ".join(self.output_root.run_advisories)
            raise OutputMutationBlocked(
                f"{action} blocked: run state is "
                f"{self.output_root.run_completion}. {detail}"
            )
        if not self.output_root.has_exhaustive_processing_inventory:
            raise OutputMutationBlocked(
                f"{action} blocked: this read-only binding does not carry an "
                "exhaustive processing inventory. Refresh Results and Analysis."
            )

        # Check completion evidence on both sides of the exhaustive inventory
        # verification. This closes the mutation receipt over owner/manifest
        # changes without making read-only bindings walk unrelated artifacts.
        fresh_state = resolve_output_run_state(
            self.output_root.layout, depth="deep"
        )
        fresh_digest = _identity_digest(fresh_state)
        if not _run_is_complete(fresh_state):
            detail = "; ".join(_advisories(fresh_state))
            raise OutputMutationBlocked(
                f"{action} blocked: run state is "
                f"{_completion(fresh_state)}. {detail}"
            )
        if fresh_digest != _identity_digest(self.output_root.run_state):
            raise OutputMutationBlocked(
                f"{action} blocked: the run identity changed after this "
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
        verified_state = resolve_output_run_state(
            self.output_root.layout, depth="deep"
        )
        # `has_active_owner` was `owner_status in {...}` with NO liveness
        # probe. `active_run_is_currently_running()` is that same predicate
        # and it survives Task 2, so it is used here rather than
        # `completion == "active"`, which additionally requires a live pid or
        # a SLURM lifecycle record and would therefore stop tripping on the
        # pid-less owner records a SLURM launch writes.
        if (
            not _run_is_complete(verified_state)
            or _identity_digest(verified_state) != fresh_digest
            or self.output_root.active_run_is_currently_running()
        ):
            raise OutputMutationBlocked(
                f"{action} blocked: the run changed while processing "
                "artifacts were verified. Refresh Results and Analysis."
            )
        return OutputMutationReceipt(
            action=action,
            binding_generation=self.binding_generation,
            processing_fingerprint=(
                self.output_root.snapshot.processing_fingerprint
            ),
            run_identity_digest=fresh_digest,
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
    return not output_root.run_is_complete or output_root.snapshot.active_run


def output_read_only_diagnostic(output_root: OutputRoot) -> str | None:
    """Return a visible diagnostic for a browsable but non-mutable output."""
    if not output_mutations_disabled(output_root):
        return None
    reasons = "; ".join(output_root.run_advisories)
    return (
        f"Read-only output: run state is "
        f"{output_root.run_completion}. {reasons} Persistent QC, "
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
