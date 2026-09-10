"""Presentation model for the shared Results/Analysis binding hand-off."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["BindingUiState", "binding_error_text", "binding_ui_state"]

_AUTHORITATIVE_ACTIVE_STATUSES = frozenset({"queued", "running"})
_WORKING_STATUSES = _AUTHORITATIVE_ACTIVE_STATUSES | {"submitting"}
_PHASE_LABELS = {
    "submitting": "Submitting request",
    "queued": "Queued",
    "classifying": "Classifying output",
    "inventory": "Scanning processing inventory",
    "measurements": "Loading measurements",
    "indexing": "Indexing viewer data",
    "verifying": "Verifying snapshot",
    "building_results": "Building Results",
    "building_analysis": "Building Analysis",
    "publishing": "Publishing both sessions",
    "complete": "Publication complete",
    "failed": "Binding failed",
    "cancelled": "Binding cancelled",
    "superseded": "Binding superseded",
    "submission_unknown": "Submission acknowledgement unavailable",
}
_STATUS_LABELS = {
    "submitting": "Submitting",
    "queued": "Queued",
    "running": "Active",
    "succeeded": "Published",
    "failed": "Failed",
    "cancelled": "Cancelled",
    "superseded": "Superseded",
    "unknown": "Unconfirmed",
}


@dataclass(frozen=True)
class BindingUiState:
    """Render-ready state for the shell binding card."""

    panel_class_name: str
    status: str
    phase: str
    detail: str
    progress_value: int | None
    progress_max: int
    progress_label: str
    diagnostic: str
    cancel_disabled: bool
    poll_disabled: bool


def binding_ui_state(payload: object) -> BindingUiState:
    """Translate one session-store payload into the sidebar presentation."""
    if not isinstance(payload, Mapping):
        return BindingUiState(
            panel_class_name=(
                "shell-results-binding-panel "
                "shell-results-binding-panel--hidden"
            ),
            status="Idle",
            phase="",
            detail="",
            progress_value=0,
            progress_max=1,
            progress_label="",
            diagnostic="",
            cancel_disabled=True,
            poll_disabled=True,
        )

    job_value = payload.get("job")
    job = job_value if isinstance(job_value, Mapping) else payload
    status = str(job.get("status") or payload.get("status") or "failed")
    phase_key = str(job.get("phase") or status)
    completed = _optional_nonnegative_int(job.get("completed"))
    total = _optional_positive_int(job.get("total"))
    authoritative_active = status in _AUTHORITATIVE_ACTIVE_STATUSES
    working = status in _WORKING_STATUSES

    if total is not None:
        progress_max = total
        progress_value = min(completed or 0, total)
        progress_label = f"{progress_value} of {total}"
    elif working:
        progress_max = 1
        progress_value = None
        progress_label = "Working"
    else:
        progress_max = 1
        progress_value = 1
        progress_label = ""

    detail = str(
        job.get("detail")
        or job.get("error")
        or payload.get("error")
        or ""
    )
    target_value = job.get("target") or payload.get("abs_path")
    target = (
        Path(str(target_value)).name
        if isinstance(target_value, (str, Path)) and str(target_value)
        else "output"
    )
    diagnostic = _diagnostic_text(payload, job, status=status, target=target)
    tone = status if status in _STATUS_LABELS else "failed"
    return BindingUiState(
        panel_class_name=(
            "shell-results-binding-panel "
            f"shell-results-binding-panel--{tone}"
        ),
        status=_STATUS_LABELS.get(status, "Unavailable"),
        phase=_PHASE_LABELS.get(
            phase_key,
            phase_key.replace("_", " ").strip().title(),
        ),
        detail=detail,
        progress_value=progress_value,
        progress_max=progress_max,
        progress_label=progress_label,
        diagnostic=diagnostic,
        cancel_disabled=not authoritative_active,
        poll_disabled=not authoritative_active,
    )


def binding_error_text(payload: object) -> str:
    """Return the page-local binding error or submission warning, if any."""
    if not isinstance(payload, Mapping):
        return ""
    if payload.get("submission_outcome") == "unknown":
        detail = str(payload.get("submission_error") or "").strip()
        suffix = f" {detail}" if detail else ""
        return (
            "Binding submission could not be confirmed and may have been "
            f"accepted.{suffix}"
        )
    job_value = payload.get("job")
    job = job_value if isinstance(job_value, Mapping) else payload
    status = str(job.get("status") or payload.get("status") or "")
    if status not in {"failed", "cancelled", "superseded"}:
        return ""
    return str(
        job.get("error")
        or payload.get("error")
        or job.get("detail")
        or f"Results binding {status}."
    )


def _diagnostic_text(
    payload: Mapping[Any, Any],
    job: Mapping[Any, Any],
    *,
    status: str,
    target: str,
) -> str:
    notices: list[str] = []
    submission_pending = payload.get("submission_outcome") == "pending"
    submission_unknown = payload.get("submission_outcome") == "unknown"
    if payload.get("deduplicated") is True:
        notices.append("Reused the active request.")
    if payload.get("superseded_job_id"):
        notices.append("Superseded the previous request.")

    attempt = _optional_positive_int(job.get("attempt"))
    if attempt is not None and attempt > 1:
        notices.append(f"Stable-read attempt {attempt}.")
    if job.get("cache_hit") is True:
        notices.append("Verified inventory cache hit.")
    poll_error = payload.get("poll_error")
    if isinstance(poll_error, str) and poll_error:
        notices.append(f"Progress check unavailable: {poll_error}")
    cancel_error = payload.get("cancel_error")
    if isinstance(cancel_error, str) and cancel_error:
        notices.append(f"Cancellation not confirmed: {cancel_error}")
    if submission_unknown:
        submission_error = payload.get("submission_error")
        detail = (
            f" ({submission_error})"
            if isinstance(submission_error, str) and submission_error
            else ""
        )
        notices.append(
            "The latest submission could not be confirmed and may have been "
            f"accepted{detail}."
        )
    elif submission_pending and status in _AUTHORITATIVE_ACTIVE_STATUSES:
        notices.append(
            "A newer binding request is awaiting acknowledgement. Continuing "
            "to monitor the previously acknowledged job until the new request "
            "returns an authoritative job identifier."
        )

    if status == "succeeded":
        snapshot_value = payload.get("snapshot")
        if not isinstance(snapshot_value, Mapping):
            result_value = job.get("result")
            snapshot_value = (
                result_value.get("snapshot")
                if isinstance(result_value, Mapping)
                else None
            )
        fingerprint = (
            snapshot_value.get("processing_fingerprint")
            if isinstance(snapshot_value, Mapping)
            else None
        )
        fingerprint_text = (
            f" Snapshot {str(fingerprint)[:12]}."
            if isinstance(fingerprint, str) and fingerprint
            else ""
        )
        notices.append(
            f"{target} published atomically to Results + Analysis."
            f"{fingerprint_text}"
        )
        consistency_value = payload.get("consistency")
        if not isinstance(consistency_value, Mapping):
            result_value = job.get("result")
            consistency_value = (
                result_value.get("consistency")
                if isinstance(result_value, Mapping)
                else None
            )
        if isinstance(consistency_value, Mapping):
            consistency_state = str(
                consistency_value.get("state") or "unavailable"
            )
            reasons_value = consistency_value.get("reasons")
            reasons = (
                [str(value) for value in reasons_value]
                if isinstance(reasons_value, list)
                else []
            )
            reason_text = f" {'; '.join(reasons)}" if reasons else ""
            notices.append(
                f"Consistency: {consistency_state}.{reason_text}".strip()
            )
    elif status == "failed":
        kind = str(job.get("error_kind") or payload.get("error_kind") or "")
        if submission_unknown:
            notices.append(
                "The previously acknowledged job failed, but the "
                "unacknowledged request may still have published. The "
                "current Results + Analysis publication cannot be inferred."
            )
        elif kind == "invalid":
            notices.append(
                "Compatibility validation failed. The previous Results + "
                "Analysis publication is unchanged."
            )
        elif kind == "stale":
            notices.append(
                "Snapshot consistency changed during binding. The previous "
                "Results + Analysis publication is unchanged."
            )
        else:
            notices.append(
                "Binding is unavailable. The previous Results + Analysis "
                "publication is unchanged."
            )
    elif status == "cancelled":
        if submission_unknown:
            notices.append(
                "The previously acknowledged job was cancelled, but the "
                "unacknowledged request may still have published. The "
                "current Results + Analysis publication cannot be inferred."
            )
        else:
            notices.append(
                "Cancelled before publication. The previous Results + "
                "Analysis publication is unchanged."
            )
    elif status == "superseded":
        notices.append(
            "A newer request won. The superseded candidate was not published."
        )
    elif status == "submitting":
        notices.append(
            "Waiting for an authoritative job identifier before polling or "
            "cancellation becomes available."
        )
    elif status in _AUTHORITATIVE_ACTIVE_STATUSES:
        if submission_unknown:
            notices.append(
                "Continuing to monitor the previously acknowledged job. Its "
                "progress does not establish whether the unacknowledged "
                "request published."
            )
        else:
            notices.append(
                f"Preparing {target}; publication has not changed yet."
            )

    return " ".join(notices)


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _optional_positive_int(value: Any) -> int | None:
    value_int = _optional_nonnegative_int(value)
    return value_int if value_int is not None and value_int > 0 else None
