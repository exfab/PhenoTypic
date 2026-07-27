"""Read-only classification of Results output completion evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from phenotypic.sdk_ import (
    BundleLayout,
    DashboardManifestKey,
    ProcessingStateKey,
    RUN_COMPLETION_JSON,
    gui_launch_owner_path,
    resolve_manifest_json_path,
    resolve_progress_dir,
    resolve_processing_state_path,
)

OutputConsistencyState = Literal[
    "coherent",
    "active",
    "incomplete",
    "contradictory",
]

_ACTIVE_OWNER_STATUSES = frozenset(
    {"queued", "submitting", "running", "reconciling", "cancelling", "unknown"}
)
_TERMINAL_OWNER_STATUSES = frozenset({"complete", "failed", "cancelled"})
_STAGED_COMPLETION_FILENAME = "staged_finalization_complete.json"
_STAGED_ORCHESTRATION_FILENAME = "staged_orchestration.json"


@dataclass(frozen=True)
class OutputCompletionEvidence:
    """Normalized completion evidence read from one output."""

    standalone_bundle: bool
    owner_present: bool = False
    owner_readable: bool = True
    owner_status: str | None = None
    manifest_present: bool = False
    manifest_readable: bool = True
    manifest_is_complete: bool | None = None
    manifest_completed: int | None = None
    manifest_failed: int | None = None
    manifest_total: int | None = None
    completion_marker_present: bool = False
    completion_marker_valid: bool = False
    staged_marker_present: bool = False
    staged_marker_valid: bool = False
    processing_state_present: bool = False
    processing_state_readable: bool = True
    processing_total: int | None = None
    processing_completed: int | None = None
    processing_failed: int | None = None
    processing_unfinished: int | None = None


@dataclass(frozen=True)
class OutputConsistencyReport:
    """Pure classification of normalized output completion evidence.

    Contradictory and incomplete outputs remain valid read-only discovery
    targets. Only ``coherent`` terminal outputs are eligible for persistent
    processing-inventory cache reuse or later mutation authorization.
    """

    state: OutputConsistencyState
    reasons: tuple[str, ...]
    evidence: OutputCompletionEvidence
    evidence_fingerprint: str

    @property
    def is_read_only(self) -> bool:
        """Return whether downstream consumers must prohibit mutations."""
        return self.state != "coherent"

    @property
    def cache_reusable(self) -> bool:
        """Return whether immutable processing inventory may be reused."""
        return self.state == "coherent"

    @property
    def has_active_owner(self) -> bool:
        """Return whether a nonterminal GUI owner was observed."""
        return self.evidence.owner_status in _ACTIVE_OWNER_STATUSES


def classify_output_consistency(
    evidence: OutputCompletionEvidence,
) -> OutputConsistencyReport:
    """Classify already-normalized evidence without filesystem access.

    Args:
        evidence: Normalized evidence captured by a read-only inspector.

    Returns:
        Immutable consistency report suitable for Results and Analysis.
    """
    contradictions: list[str] = []
    incompleteness: list[str] = []

    owner_active = evidence.owner_status in _ACTIVE_OWNER_STATUSES
    owner_success = evidence.owner_status == "complete"
    owner_failed = evidence.owner_status in {"failed", "cancelled"}
    owner_status_known = evidence.owner_status in (
        _ACTIVE_OWNER_STATUSES | _TERMINAL_OWNER_STATUSES
    )

    manifest_counts_valid = all(
        value is not None and value >= 0
        for value in (
            evidence.manifest_completed,
            evidence.manifest_failed,
            evidence.manifest_total,
        )
    )
    manifest_success = (
        evidence.manifest_readable
        and evidence.manifest_is_complete is True
        and manifest_counts_valid
        and evidence.manifest_failed == 0
        and evidence.manifest_completed == evidence.manifest_total
    )
    manifest_explicitly_incomplete = (
        evidence.manifest_present
        and evidence.manifest_readable
        and (
            evidence.manifest_is_complete is False
            or (
                manifest_counts_valid
                and evidence.manifest_completed != evidence.manifest_total
            )
            or (
                evidence.manifest_failed is not None
                and evidence.manifest_failed > 0
            )
        )
    )
    terminal_marker = (
        evidence.completion_marker_valid or evidence.staged_marker_valid
    )

    if evidence.manifest_present and not evidence.manifest_readable:
        incompleteness.append("publication manifest is unreadable")
    if evidence.owner_present and not evidence.owner_readable:
        incompleteness.append("output owner record is unreadable")
    elif evidence.owner_present and not owner_status_known:
        incompleteness.append("output owner status is missing or unknown")
    if evidence.processing_state_present and not evidence.processing_state_readable:
        incompleteness.append("processing state is unreadable")
    if evidence.completion_marker_present and not evidence.completion_marker_valid:
        incompleteness.append("ordinary completion marker is invalid")
    if evidence.staged_marker_present and not evidence.staged_marker_valid:
        incompleteness.append("staged completion marker is invalid")

    if manifest_counts_valid:
        completed = evidence.manifest_completed
        failed = evidence.manifest_failed
        total = evidence.manifest_total
        assert completed is not None
        assert failed is not None
        assert total is not None
        if completed > total:
            contradictions.append(
                f"manifest completed count exceeds total ({completed}>{total})"
            )
        if completed + failed > total:
            contradictions.append(
                "manifest completed plus failed counts exceed total "
                f"({completed}+{failed}>{total})"
            )

    if owner_active and (terminal_marker or manifest_success):
        contradictions.append(
            "active owner conflicts with successful terminal publication"
        )
    if owner_failed and (terminal_marker or manifest_success):
        contradictions.append(
            f"owner status {evidence.owner_status!r} conflicts with "
            "successful terminal publication"
        )
    if terminal_marker and manifest_explicitly_incomplete:
        contradictions.append(
            "completion marker conflicts with incomplete or failed manifest"
        )

    if evidence.processing_state_readable and evidence.processing_total is not None:
        state_total = evidence.processing_total
        state_completed = evidence.processing_completed
        state_failed = evidence.processing_failed
        if (
            state_completed is not None
            and state_failed is not None
            and state_completed + state_failed > state_total
        ):
            contradictions.append(
                "processing-state completed plus failed counts exceed inventory "
                f"({state_completed}+{state_failed}>{state_total})"
            )
        if manifest_counts_valid and evidence.manifest_total != state_total:
            contradictions.append(
                "manifest total conflicts with processing inventory "
                f"({evidence.manifest_total}!={state_total})"
            )
        if (
            manifest_counts_valid
            and state_completed is not None
            and evidence.manifest_completed != state_completed
        ):
            contradictions.append(
                "manifest completed count conflicts with processing state "
                f"({evidence.manifest_completed}!={state_completed})"
            )

    if contradictions:
        state: OutputConsistencyState = "contradictory"
        reasons = tuple(dict.fromkeys(contradictions + incompleteness))
    elif owner_active:
        state = "active"
        reasons = ("a nonterminal GUI owner is active",)
    elif evidence.standalone_bundle and not incompleteness:
        state = "coherent"
        reasons = ("standalone deliverables bundle is a terminal snapshot",)
    elif manifest_success and not owner_failed and not incompleteness:
        state = "coherent"
        reasons = ("terminal manifest evidence is internally coherent",)
    elif owner_success and not evidence.manifest_present:
        state = "incomplete"
        reasons = (
            "terminal owner record has no successful publication manifest",
        )
    else:
        if manifest_explicitly_incomplete:
            incompleteness.append("manifest reports incomplete or failed work")
        if not evidence.manifest_present:
            incompleteness.append("no terminal publication manifest exists")
        if evidence.processing_unfinished:
            incompleteness.append(
                f"processing inventory has {evidence.processing_unfinished} "
                "unfinished image(s)"
            )
        state = "incomplete"
        reasons = tuple(dict.fromkeys(incompleteness)) or (
            "no coherent terminal evidence exists",
        )

    serialized = json.dumps(
        asdict(evidence),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    evidence_fingerprint = f"sha256:{hashlib.sha256(serialized).hexdigest()}"
    return OutputConsistencyReport(
        state=state,
        reasons=reasons,
        evidence=evidence,
        evidence_fingerprint=evidence_fingerprint,
    )


def inspect_output_consistency(layout: BundleLayout) -> OutputConsistencyReport:
    """Read completion sidecars without modifying the selected output.

    Args:
        layout: Detected output or standalone-bundle topology.

    Returns:
        Pure consistency report derived from normalized evidence.
    """
    if layout.output_root is None:
        return classify_output_consistency(
            OutputCompletionEvidence(standalone_bundle=True)
        )

    output_root = layout.output_root
    resolved_progress = resolve_progress_dir(output_root)
    owner_path = gui_launch_owner_path(output_root)
    owner_payload, owner_readable = _read_json(owner_path)
    manifest_path = resolve_manifest_json_path(output_root)
    manifest_payload, manifest_readable = _read_json(manifest_path)
    completion_path = resolved_progress / RUN_COMPLETION_JSON
    completion_payload, _ = _read_json(completion_path)
    staged_path = resolved_progress / _STAGED_COMPLETION_FILENAME
    staged_payload, _ = _read_json(staged_path)
    orchestration_payload, _ = _read_json(
        resolved_progress / _STAGED_ORCHESTRATION_FILENAME
    )
    processing_path = resolve_processing_state_path(output_root)
    processing_payload, processing_readable = _read_json(processing_path)

    processing_counts = _processing_counts(processing_payload)
    staged_valid = _staged_marker_is_valid(
        staged_payload,
        orchestration_payload=orchestration_payload,
    )
    evidence = OutputCompletionEvidence(
        standalone_bundle=False,
        owner_present=owner_path.is_file(),
        owner_readable=owner_readable,
        owner_status=_string_value(owner_payload, "status"),
        manifest_present=manifest_path.is_file(),
        manifest_readable=manifest_readable,
        manifest_is_complete=_bool_value(
            manifest_payload,
            DashboardManifestKey.IS_COMPLETE,
        ),
        manifest_completed=_int_value(
            manifest_payload,
            DashboardManifestKey.COMPLETED,
        ),
        manifest_failed=_int_value(
            manifest_payload,
            DashboardManifestKey.FAILED,
        ),
        manifest_total=_int_value(
            manifest_payload,
            DashboardManifestKey.TOTAL_IMAGES,
        ),
        completion_marker_present=completion_path.is_file(),
        completion_marker_valid=(
            _string_value(completion_payload, "status") == "complete"
            and completion_payload.get("finalizer_succeeded") is True
            if completion_payload is not None
            else False
        ),
        staged_marker_present=staged_path.is_file(),
        staged_marker_valid=staged_valid,
        processing_state_present=processing_path.is_file(),
        processing_state_readable=processing_readable,
        processing_total=processing_counts[0],
        processing_completed=processing_counts[1],
        processing_failed=processing_counts[2],
        processing_unfinished=processing_counts[3],
    )
    return classify_output_consistency(evidence)


def _read_json(path: Path) -> tuple[dict[str, object] | None, bool]:
    """Read one JSON object, returning a readability flag."""
    if not path.is_file():
        return None, True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, False
    return (payload, True) if isinstance(payload, dict) else (None, False)


def _string_value(
    payload: dict[str, object] | None,
    key: str,
) -> str | None:
    value = payload.get(key) if payload is not None else None
    return value if isinstance(value, str) else None


def _bool_value(
    payload: dict[str, object] | None,
    key: str,
) -> bool | None:
    value = payload.get(key) if payload is not None else None
    return value if isinstance(value, bool) else None


def _int_value(
    payload: dict[str, object] | None,
    key: str,
) -> int | None:
    value = payload.get(key) if payload is not None else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _processing_counts(
    payload: dict[str, object] | None,
) -> tuple[int | None, int | None, int | None, int | None]:
    """Return total, completed, failed, and unfinished state counts."""
    if payload is None:
        return (None, None, None, None)
    raw_datasets = payload.get(ProcessingStateKey.DATASETS)
    if not isinstance(raw_datasets, dict):
        return (None, None, None, None)
    total = completed = failed = 0
    for raw_state in raw_datasets.values():
        if not isinstance(raw_state, dict):
            return (None, None, None, None)
        initial = _string_list(raw_state.get(ProcessingStateKey.INITIAL_IMAGES))
        completed_images = _string_list(
            raw_state.get(ProcessingStateKey.COMPLETED)
        )
        failed_images = _string_list(raw_state.get(ProcessingStateKey.FAILED))
        if initial is None or completed_images is None or failed_images is None:
            return (None, None, None, None)
        total += len(initial)
        completed += len(initial & completed_images)
        failed += len(initial & failed_images)
    return (total, completed, failed, max(total - completed - failed, 0))


def _string_list(value: object) -> set[str] | None:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        return None
    return set(value)


def _staged_marker_is_valid(
    marker: dict[str, object] | None,
    *,
    orchestration_payload: dict[str, object] | None,
) -> bool:
    if marker is None or not isinstance(marker.get("completed_at"), str):
        return False
    mode = marker.get("mode")
    if mode == "local":
        return isinstance(marker.get("pipeline_sha256"), str)
    epoch = marker.get("epoch")
    if not isinstance(epoch, str) or not epoch:
        return False
    if orchestration_payload is None:
        return True
    return (
        orchestration_payload.get("epoch") == epoch
        and orchestration_payload.get("phase") == "complete"
    )
