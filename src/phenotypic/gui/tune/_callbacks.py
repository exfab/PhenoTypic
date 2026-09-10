"""Dash callbacks for the ``/tune/`` co-pilot.

Three concerns live here, each a thin Dash adapter around a pure, headless-
testable helper:

* **Run binding** — :func:`~phenotypic.gui.tune._run_picker.discover_run_payload`
  validates a sandbox-bounded directory as a tune output; the bind callback
  writes the discovered run-root payload into ``TUNE_RUN_ROOT_STORE`` and swaps
  the page body to the loaded four-view layout (or surfaces a note on failure).
* **Sub-tab switching** — :func:`active_view` maps a clicked sub-tab button's
  ID to its view name (falling back to the default Monitor view for an unknown
  or absent trigger); the registered callback toggles which view container is
  visible and which button carries the active class.
* **Monitor poll** — :func:`read_study_for_monitor` reads the run's study every
  poll tick (live ``OptunaStudyStore`` when the ``tune`` extra is importable AND
  the storage URL connects within a short timeout, else the finished
  ``trials.parquet``); the registered poll callback re-builds the objective /
  importance figures, the gap badge, and the trials table from it.

The module never imports ``optuna`` at import time. The live store is imported
**inside** :func:`read_study_for_monitor` (gated on
``importlib.util.find_spec("optuna")``), so this module stays in the package's
optuna-free import surface.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import TYPE_CHECKING, Any, Callable, Optional

from dash import ctx, no_update

from phenotypic.gui._config import (
    CFG_RUNNER,
    CFG_RUN_REGISTRY,
    IMAGE_EXTS,
    THREAD_NAME_PREFIX,
)
from phenotypic.gui.shell._ids import (
    SHELL_METADATA_CSV_STORE,
    SHELL_SOURCE_IMAGE_ROOT_STORE,
    TUNE_PIPELINE_PATH_STORE,
)
from phenotypic.gui.shell._source_context import (
    SourcePayload,
    resolve_source_image_root,
    source_payload_from_path,
)
from phenotypic.gui.tune import _ids as ids
from phenotypic.gui.tune import _nav
from phenotypic.gui.tune._command import (
    DEFAULT_STORAGE_ENV,
    StorageMode,
    build_tune_command,
    storage_url_preflight_issue,
)
from phenotypic.gui.tune._deploy import deploy_tune_run
from phenotypic.gui.tune._export import (
    prepare_best_from_run,
    publish_prepared_export,
)
from phenotypic.gui.tune._monitor import (
    cancel_prompt,
    parse_run_receipt,
    run_switcher_items,
)
from phenotypic.gui.tune._run_image_source import resolve_run_images
from phenotypic.gui.tune._setup_authoring import (
    SetupDraft,
    SetupDraftCache,
    SetupPathResolution,
    SetupWriteReceipt,
    authored_content_fingerprint,
    build_setup_draft,
    load_pipeline_or_spec,
    path_content_fingerprint,
    resolve_setup_path,
    setup_draft_from_store,
    setup_path_resolution_from_store,
    setup_path_payload,
    write_setup_draft_receipt,
)
from phenotypic.gui.tune._validation import preflight_issues, spec_path_issue
from phenotypic.sdk_ import (
    CONFIG_SUFFIX_TUNING,
    PIPELINE_CONFIG_SUFFIXES,
    matches_any_suffix,
)

if TYPE_CHECKING:
    from pathlib import Path

    from phenotypic.gui.shell._sandbox import SandboxRoot
    from phenotypic.gui.tune._study_read import _ReadableStore
    from phenotypic.gui.tune._run_root import TuneRunRoot
    from phenotypic.tune._study_store import Trial

logger = logging.getLogger(__name__)

#: The default view shown when no (or an unknown) sub-tab is active.
_DEFAULT_VIEW: ids.SubTabName = "monitor"

#: Hard cap on the open and every storage read used by a Monitor poll.
_LIVE_READ_TIMEOUT_S: float = 3.0

#: Driver-level Postgres connect bound nested inside the whole-read deadline.
_LIVE_CONNECT_TIMEOUT_S: float = 2.0

#: Schemes (SQLAlchemy backend names) whose driver honors a libpq-style
#: ``connect_timeout`` query param. SQLite is local (no network hang), so it is
#: deliberately absent — :func:`_ensure_connect_timeout` no-ops on it.
_CONNECT_TIMEOUT_BACKENDS: frozenset[str] = frozenset({"postgresql"})

#: A process-wide, single-worker pool for the live-study open. Shared (not a
#: per-call ``with`` block) so a still-connecting worker is NEVER re-joined:
#: when :func:`read_study_for_monitor` times out it returns the parquet
#: fallback immediately and leaves the orphaned worker to finish (and be
#: discarded) on its own. :data:`_LIVE_READS` coalesces repeated polls onto the
#: same pending future and refuses a different run until that one finishes, so
#: no queued fan of connect attempts can accumulate against a dead host.
_LIVE_OPEN_POOL: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix=f"{THREAD_NAME_PREFIX}-tune-live-open"
)


class _LiveReadCoalescer:
    """Keep at most one live-study read submitted process-wide."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: Optional[tuple[str, str]] = None
        self._pending: "Optional[Future[_ReadableStore]]" = None

    def acquire(
        self,
        key: tuple[str, str],
        submit: "Callable[[], Future[_ReadableStore]]",
    ) -> "Optional[Future[_ReadableStore]]":
        """Return the matching pending read or submit one when the slot is free."""
        with self._lock:
            if self._pending is not None:
                if not self._pending.done():
                    return self._pending if self._key == key else None
                if self._key == key:
                    return self._pending
                self._pending = None
                self._key = None
            self._key = key
            self._pending = submit()
            return self._pending

    def consume(self, future: "Future[_ReadableStore]") -> None:
        """Release a completed read after its result has been observed."""
        with self._lock:
            if self._pending is future:
                self._pending = None
                self._key = None

    def clear(self) -> None:
        """Forget a completed read, for deterministic test isolation."""
        with self._lock:
            if self._pending is None or self._pending.done():
                self._pending = None
                self._key = None


_LIVE_READS: _LiveReadCoalescer = _LiveReadCoalescer()

#: fANOVA is slower than the poll deadline and must never backlog storage reads.
_LIVE_IMPORTANCE_POOL: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix=f"{THREAD_NAME_PREFIX}-tune-importances"
)

#: The degrade note shown when a live read was attempted but the storage could
#: not be reached in time — points the user at the usual culprits.
_NOTE_LIVE_UNREACHABLE: str = (
    "couldn't reach the live study -- check network / ~/.pgpass. "
    "Showing the last finished trials."
)

#: The note for a pre-cutover ("tune", maximize) run reached after the cost
#: cutover: its study can't be opened by the new name, and even if it could the
#: stored cost/score meaning is inverted, so the only safe action is a re-run.
_NOTE_LEGACY_STUDY: str = (
    "this run predates the cost cutover (study 'tune') -- its results use the "
    "old higher-is-better convention and can't be monitored live. Re-run it "
    "with the current phenotypic.tune to get cost-convention results."
)


def _source_payload_for_tune_image_source(
    sandbox: "SandboxRoot",
    image_source: Optional[str],
    current_payload: object,
) -> SourcePayload | None:
    """Build a shared source payload from Tune's Image Source store."""
    if not image_source:
        return None
    payload = source_payload_from_path(sandbox, image_source, source="tune")
    if payload is None:
        return None
    if (
        isinstance(current_payload, dict)
        and current_payload.get("abs_path") == payload["abs_path"]
    ):
        return None
    return payload


def _tune_image_source_from_shared(
    sandbox: "SandboxRoot",
    shared_payload: object,
    current_image_source: Optional[str],
) -> str | None:
    """Return a Tune Image Source from shared source when Tune is unset."""
    if current_image_source:
        return None
    resolved = resolve_source_image_root(sandbox, shared_payload)
    return str(resolved) if resolved is not None else None

#: The note shown when the run wants a live study but the ``tune`` extra (and so
#: ``optuna``) is not installed.
_NOTE_MISSING_EXTRA: str = (
    "install the tune extra for live monitoring "
    "(uv sync --extra tune)."
)

#: Reverse map: a sub-tab button ID -> its view name. Built once from the
#: ordered sub-tab names so the helper never re-derives strings at call time.
_BUTTON_ID_TO_VIEW: dict[str, ids.SubTabName] = {
    ids.subtab_button_id(name): name for name in ids.SUBTAB_ORDER
}


def setup_gate_state(
    pipeline_path: str | None,
    metadata_path: str | None = None,
    *,
    source_is_spec: bool = False,
    replace_scorer: bool = False,
    validation_issues: tuple[str, ...] = (),
) -> tuple[str, str, bool, str]:
    """Return Setup section classes, Continue state, and complete gate note."""
    if pipeline_path:
        metadata_required = not source_is_spec or replace_scorer
        if (metadata_path or not metadata_required) and not validation_issues:
            metadata_note = (
                f" Metadata selected: {metadata_path}."
                if metadata_path
                else " Existing scorer preserved."
            )
            return (
                "tune-setup-section",
                "tune-setup-section",
                False,
                f"Pipeline selected: {pipeline_path}.{metadata_note}",
            )
        issue_text = " ".join(validation_issues)
        if issue_text:
            note = issue_text
        else:
            note = (
                f"Pipeline selected: {pipeline_path}. "
                "Add metadata to author a tune spec."
            )
        return (
            "tune-setup-section",
            "tune-setup-section",
            True,
            note,
        )
    return (
        "tune-setup-section tune-setup-locked",
        "tune-setup-section tune-setup-locked",
        True,
        "Choose a pipeline and metadata layout to author a tune spec.",
    )


def setup_pipeline_path_from_sources(
    typed_path: str | None,
    shell_handoff: object,
    picker_path: str | None = None,
    *,
    sandbox: "SandboxRoot | None" = None,
) -> str | None:
    """Resolve pipeline text using typed, picker, then handoff precedence."""
    candidates: list[str] = []
    if typed_path:
        candidates.append(typed_path)
    if picker_path:
        candidates.append(picker_path)
    if sandbox is not None:
        shared = resolve_setup_path(
            sandbox=sandbox,
            kind="pipeline",
            typed_path=None,
            picker_payload=None,
            shared_payload=shell_handoff,
        )
        if shared.path is not None:
            candidates.append(str(shared.path))

    valid_suffixes = PIPELINE_CONFIG_SUFFIXES | frozenset({CONFIG_SUFFIX_TUNING})
    for candidate in candidates:
        path = candidate.strip()
        if path and matches_any_suffix(path, valid_suffixes):
            return path
    return None


def authored_spec_descriptor(
    *,
    path: str,
    pipeline_path: str,
    metadata_path: str | None,
    replace_scorer: bool = False,
    setup_signature: str | None = None,
    launch_defaults: dict[str, object] | None = None,
    write_receipt: SetupWriteReceipt | None = None,
) -> dict[str, object]:
    """Return a descriptor bound to the exact draft write provenance."""
    source_path = Path(pipeline_path)
    selected_metadata = Path(metadata_path) if metadata_path else None
    authored_path = Path(path)
    if write_receipt is not None:
        if (
            write_receipt.path != authored_path
            or write_receipt.draft_revision != (setup_signature or "")
        ):
            raise ValueError("write receipt does not match the authored Setup draft")
        source_fingerprint = write_receipt.source_fingerprint
        metadata_fingerprint = write_receipt.metadata_fingerprint
        authored_fingerprint = write_receipt.authored_fingerprint
    else:
        source_fingerprint = path_content_fingerprint(source_path)
        metadata_fingerprint = path_content_fingerprint(selected_metadata)
        authored_fingerprint = authored_content_fingerprint(authored_path)
    return {
        "version": 2,
        "path": path,
        "pipeline_path": pipeline_path,
        "metadata_path": metadata_path or "",
        "replace_scorer": replace_scorer,
        "setup_signature": setup_signature or "",
        "setup_revision": setup_signature or "",
        "source_fingerprint": source_fingerprint,
        "metadata_fingerprint": metadata_fingerprint,
        "authored_fingerprint": authored_fingerprint,
        "launch_defaults": dict(launch_defaults or {}),
    }


def active_authored_spec_path(
    descriptor: object,
    *,
    pipeline_path: str | None,
    metadata_path: str | None,
    replace_scorer: bool = False,
    setup_signature: str | None = None,
) -> str | None:
    """Return the authored spec path only when it matches current Setup inputs."""
    if not isinstance(descriptor, dict):
        return None
    path = descriptor.get("path")
    source = descriptor.get("pipeline_path")
    metadata = descriptor.get("metadata_path")
    replace = descriptor.get("replace_scorer", False)
    stored_signature = descriptor.get("setup_signature", "")
    stored_source_fingerprint = descriptor.get("source_fingerprint", "")
    stored_metadata_fingerprint = descriptor.get("metadata_fingerprint", "")
    stored_authored_fingerprint = descriptor.get("authored_fingerprint", "")
    source_path = Path(pipeline_path) if pipeline_path else None
    selected_metadata = Path(metadata_path) if metadata_path else None
    if (
        isinstance(path, str)
        and path
        and isinstance(source, str)
        and isinstance(metadata, str)
        and source == (pipeline_path or "")
        and metadata == (metadata_path or "")
        and replace is replace_scorer
        and source_path is not None
        and stored_source_fingerprint
        == path_content_fingerprint(source_path)
        and stored_metadata_fingerprint
        == path_content_fingerprint(selected_metadata)
        and stored_authored_fingerprint
        == authored_content_fingerprint(Path(path))
        and (
            setup_signature is None
            or stored_signature == setup_signature
        )
    ):
        return path
    return None


def active_authored_spec_for_draft(
    descriptor: object,
    draft: SetupDraft | None,
) -> str | None:
    """Return the authored spec only when it exactly matches ``draft``."""
    if draft is None:
        return None
    return active_authored_spec_path(
        descriptor,
        pipeline_path=draft.pipeline_path,
        metadata_path=draft.metadata_path,
        replace_scorer=draft.replace_scorer,
        setup_signature=draft.revision,
    )


def setup_authoring_signature(
    *,
    pipeline_path: str | None,
    metadata_path: str | None,
    replace_scorer: bool,
    edits: dict[str, dict],
) -> str:
    """Return a stable digest for every input that affects the authored spec."""
    payload = {
        "pipeline_path": (
            str(Path(pipeline_path).expanduser().resolve(strict=False))
            if pipeline_path
            else ""
        ),
        "pipeline_fingerprint": path_content_fingerprint(
            Path(pipeline_path) if pipeline_path else None
        ),
        "metadata_path": (
            str(Path(metadata_path).expanduser().resolve(strict=False))
            if metadata_path
            else ""
        ),
        "metadata_fingerprint": path_content_fingerprint(
            Path(metadata_path) if metadata_path else None
        ),
        "replace_scorer": replace_scorer,
        "edits": edits,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _strategy_name(strategy: object) -> str:
    """Return the CLI strategy name represented by one strategy config."""
    kind = getattr(strategy, "kind", None)
    if kind == "optuna":
        sampler = getattr(strategy, "sampler", None)
        return str(sampler or "tpe")
    return str(kind or "")


def authored_spec_launch_defaults(path: Path) -> dict[str, object]:
    """Return non-secret Run control defaults from an authored spec."""
    from phenotypic.tune import TuningSpec

    spec = TuningSpec.model_validate_json(path.read_text(encoding="utf-8"))
    strategy = spec.strategy
    return {
        "strategy": _strategy_name(strategy),
        "n_trials": getattr(strategy, "n_trials", None),
        "seed": getattr(strategy, "seed", None),
        "has_spec_storage": bool(getattr(strategy, "storage_url", None)),
    }


def _descriptor_launch_defaults(descriptor: object) -> dict[str, object]:
    """Return validated launch-default metadata from an authored descriptor."""
    if not isinstance(descriptor, dict):
        return {}
    defaults = descriptor.get("launch_defaults")
    return dict(defaults) if isinstance(defaults, dict) else {}


def _path_from_resolution_store(value: object) -> str | None:
    """Read a resolved path from a Setup resolution store."""
    if not isinstance(value, dict):
        return None
    path = value.get("path")
    return path if isinstance(path, str) and path else None


def _source_label(
    label: str,
    resolution: SetupPathResolution,
) -> str:
    """Describe a Setup path and its effective precedence source."""
    if resolution.path is None:
        return f"{label}: unset"
    return f"{label} ({resolution.source}): {resolution.path}"


def _toggle_on(values: object) -> bool:
    """Return whether a one-option Checklist carries ``on``."""
    return isinstance(values, list) and "on" in values


def _optional_int(value: object) -> int | None:
    """Normalize a Dash numeric input to ``int | None``."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float, str)):
        return int(value)
    raise TypeError(f"expected numeric input, got {type(value).__name__}")


def _optional_float(value: object) -> float | None:
    """Normalize a Dash numeric input to ``float | None``."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"expected numeric input, got {type(value).__name__}")


def _build_command_from_controls(
    *,
    sandbox: "SandboxRoot",
    authored_descriptor: object,
    pipeline_store: object,
    metadata_store: object,
    replace_values: object,
    setup_signature: str | None,
    shared_source: object,
    images_override: str | None,
    output_dir: str | None,
    strategy: str | None,
    n_trials: object,
    storage_mode: str | None,
    storage_local_path: str | None,
    storage_environment_name: str | None,
    n_workers: object,
    slurm_partition: str | None,
    slurm_mem: str | None,
    slurm_time: str | None,
    held_out_fraction: object,
    cv_group: str | None,
    mode: str | None,
    screen_values: object,
    setup_draft_store: object = None,
    setup_draft_cache: SetupDraftCache | None = None,
):  # type: ignore[no-untyped-def]
    """Build the authoritative command from the raw action controls."""
    draft = (
        setup_draft_from_store(setup_draft_store, cache=setup_draft_cache)
        if setup_draft_cache is not None
        else None
    )
    if draft is not None:
        spec_path = active_authored_spec_for_draft(authored_descriptor, draft)
    else:
        pipeline_path = _path_from_resolution_store(pipeline_store)
        metadata_path = _path_from_resolution_store(metadata_store)
        replace = _toggle_on(replace_values)
        spec_path = active_authored_spec_path(
            authored_descriptor,
            pipeline_path=pipeline_path,
            metadata_path=metadata_path,
            replace_scorer=replace,
            setup_signature=setup_signature,
        )
    issues: list[str] = []
    defaults = _descriptor_launch_defaults(authored_descriptor)
    default_strategy = str(defaults.get("strategy") or "")
    default_trials = defaults.get("n_trials")
    selected_strategy = (strategy or "").strip()
    spec_issue = spec_path_issue(spec_path)
    if spec_issue is not None:
        issues.append(spec_issue.message)
    elif spec_path is not None:
        try:
            issues.extend(
                _load_spec_preflight_issues(
                    spec_path, selected_strategy or default_strategy
                )
            )
        except Exception:  # noqa: BLE001
            logger.warning("Tune command preflight failed", exc_info=True)
            issues.append("Could not inspect the authored tuning spec.")

    images_dir = (
        images_override
        if images_override and images_override.strip()
        else resolve_run_images(sandbox, shared_source, None)
    )
    try:
        normalized_trials = _optional_int(n_trials)
    except (TypeError, ValueError):
        normalized_trials = None
        issues.append("Trial budget must be an integer.")
    try:
        normalized_workers = _optional_int(n_workers)
    except (TypeError, ValueError):
        normalized_workers = None
        issues.append("Worker count must be an integer.")
    try:
        normalized_fraction = _optional_float(held_out_fraction)
    except (TypeError, ValueError):
        normalized_fraction = None
        issues.append("Held-out fraction must be numeric.")

    preserve_strategy = (
        bool(default_strategy) and selected_strategy == default_strategy
    )
    strategy_override = None if preserve_strategy else selected_strategy or None
    trials_override = (
        None
        if preserve_strategy and normalized_trials == default_trials
        else normalized_trials
    )

    if storage_mode == "environment":
        normalized_storage_mode: StorageMode = "environment"
    elif storage_mode == "spec":
        normalized_storage_mode = "spec"
    else:
        normalized_storage_mode = "local"
        if storage_mode not in {None, "local"}:
            issues.append("Choose a valid storage mode.")
    return build_tune_command(
        sandbox=sandbox,
        spec_path=spec_path,
        images_dir=images_dir,
        output_dir=output_dir,
        strategy=strategy_override,
        effective_strategy=selected_strategy or default_strategy,
        n_trials=trials_override,
        storage_mode=normalized_storage_mode,
        storage_local_path=storage_local_path,
        storage_environment_name=storage_environment_name,
        n_workers=normalized_workers,
        slurm_partition=slurm_partition or None,
        slurm_mem=slurm_mem or None,
        slurm_time=slurm_time or None,
        held_out_fraction=normalized_fraction,
        cv_group=cv_group or None,
        slurm=mode == "slurm",
        screen=_toggle_on(screen_values),
        additional_issues=issues,
    )


def _record_for_monitor_receipt(
    registry: object,
    receipt: object,
) -> Any | None:
    """Resolve only the registry record matching an exact Tune receipt."""
    identity = parse_run_receipt(receipt)
    if identity is None:
        return None
    run_id, generation = identity
    record = registry.get(run_id)  # type: ignore[attr-defined]
    if record is None or record.generation != generation:
        return None
    return record


def cancel_monitor_run(
    *,
    runner: object,
    registry: object,
    receipt: object,
    confirmed: bool = True,
) -> str:
    """Cancel a running Local tune run and update the registry."""
    identity = parse_run_receipt(receipt)
    if identity is None:
        return "No run selected."
    run_id, generation = identity
    record = _record_for_monitor_receipt(registry, receipt)
    if record is None:
        return f"Selected run generation is no longer current: {run_id}"
    if record.mode != "local":
        return "SLURM cancellation is not supported in v1."
    if not confirmed:
        return cancel_prompt(record.run_id, record.mode)
    reconciled = reconcile_local_run_status(
        runner=runner,
        registry=registry,
        receipt=receipt,
    )
    if reconciled in {"complete", "failed"}:
        return f"Local run already exited: {run_id} ({reconciled})."
    cancel_prompt(record.run_id, record.mode)
    stopped = runner.stop(  # type: ignore[attr-defined]
        run_id,
        generation=generation,
    )
    if not stopped:
        return f"Local run is not active: {run_id}"
    registry.compare_and_set(  # type: ignore[attr-defined]
        run_id,
        generation,
        expected_statuses={"running", "submitting"},
        status="cancelled",
    )
    return f"Cancelled Local run: {run_id}"


def reconcile_run_status(
    *,
    runner: object,
    registry: object,
    receipt: object,
) -> str | None:
    """Reap an exited runner process and mirror its final status into the registry."""
    identity = parse_run_receipt(receipt)
    if identity is None:
        return None
    run_id, generation = identity
    record = _record_for_monitor_receipt(registry, receipt)
    if (
        record is None
        or record.mode not in {"local", "slurm"}
        or record.status not in {"running", "submitting"}
    ):
        return None
    is_running = getattr(runner, "is_running", None)
    if callable(is_running) and is_running(
        run_id,
        generation=generation,
    ):
        return str(record.status)
    reap = getattr(runner, "reap", None)
    if not callable(reap):
        return None
    returncode = reap(run_id, generation=generation)
    if returncode is None:
        return None
    status = "running" if record.mode == "slurm" and returncode == 0 else (
        "complete" if returncode == 0 else "failed"
    )
    registry.compare_and_set(  # type: ignore[attr-defined]
        run_id,
        generation,
        expected_statuses={"running", "submitting"},
        status=status,
    )
    return status


def reconcile_local_run_status(
    *,
    runner: object,
    registry: object,
    receipt: object,
) -> str | None:
    """Reap an exited Local run and mirror its final status into the registry."""
    record = _record_for_monitor_receipt(registry, receipt)
    if record is None or record.mode != "local":
        return None
    return reconcile_run_status(
        runner=runner,
        registry=registry,
        receipt=receipt,
    )


def _load_spec_preflight_issues(spec_path: str, strategy: str) -> list[str]:
    """Return deploy-blocking preflight messages for ``spec_path``."""
    from phenotypic.tune import TuningSpec

    spec = TuningSpec.model_validate_json(Path(spec_path).read_text(encoding="utf-8"))
    issues = [
        issue.message
        for issue in preflight_issues(spec.search_space, strategy=strategy)
        if issue.blocks in {"deploy", "both"}
    ]
    storage_issue = storage_url_preflight_issue(
        getattr(spec.strategy, "storage_url", None)
    )
    if storage_issue is not None:
        issues.append(storage_issue)
    return issues


def export_monitor_best_pipeline(*, registry: object, receipt: object) -> Path:
    """Export the active run's best pipeline from its params sidecar."""
    identity = parse_run_receipt(receipt)
    if identity is None:
        raise ValueError("No run selected.")
    run_id, generation = identity
    record = _record_for_monitor_receipt(registry, receipt)
    if record is None:
        raise ValueError(
            f"Selected run generation is no longer current: {run_id}"
        )
    prepared = prepare_best_from_run(record.output_dir)
    published = registry.publish_if_current_generation(  # type: ignore[attr-defined]
        run_id,
        generation,
        lambda: publish_prepared_export(prepared),
    )
    if not published:
        raise ValueError(
            f"Selected run generation is no longer current: {run_id}"
        )
    return prepared.path


def active_view(trigger_id: str | None) -> ids.SubTabName:
    """Resolve which sub-tab view a click on ``trigger_id`` should show.

    Pure routing logic, unit-testable without Dash: a known sub-tab button ID
    (``tune-subtab-<name>``) resolves to its view name; ``None``, an empty
    string, or any unrecognised ID falls back to the default Monitor view (so
    the initial render and any stray trigger land somewhere valid).

    Args:
        trigger_id: The ID of the component that fired the callback (Dash's
            ``ctx.triggered_id``), or ``None`` on the initial call.

    Returns:
        The resolved view name (one of :data:`ids.SUBTAB_ORDER`).
    """
    if not trigger_id:
        return _DEFAULT_VIEW
    return _BUTTON_ID_TO_VIEW.get(trigger_id, _DEFAULT_VIEW)


# ---------------------------------------------------------------------------
# Monitor — graceful live read (OQ4)
# ---------------------------------------------------------------------------

def _load_journal(root: "TuneRunRoot") -> "Optional[_ReadableStore]":
    """Load the finished ``trials.parquet`` journal, or ``None`` when absent.

    The fall-back read path: a parquet-only run, or a live run whose study could
    not be reached. Returns ``None`` (rather than raising) when no journal has
    been written yet, so a brand-new run degrades to "no trials yet" cleanly.
    """
    from phenotypic.tune._study_store import JournalStudyStore

    trials_path = root.trials_path
    if trials_path is None or not trials_path.exists():
        return None
    try:
        return JournalStudyStore.from_parquet(trials_path)
    except Exception:  # noqa: BLE001 - poll must never raise; degrade to empty
        logger.warning("Could not load tune journal at %s", trials_path, exc_info=True)
        return None


def _ensure_connect_timeout(storage_url: str) -> str:
    """Merge a libpq ``connect_timeout`` into a postgres storage URL.

    The real fix for the OQ4 stall: the live ``OptunaStudyStore`` constructor
    connects eagerly, and libpq's default connect timeout is ~30 s (or never,
    on a black-holed host). Merging ``connect_timeout=<N>`` (N =
    :data:`_LIVE_CONNECT_TIMEOUT_S`) into the URL makes the *constructor itself*
    return fast — SQLAlchemy's psycopg dialect passes URL query params straight
    through to the driver, and libpq honors ``connect_timeout``.

    Only ``postgresql*`` schemes are touched (the backend name covers
    ``postgresql``, ``postgresql+psycopg``, ``postgresql+psycopg2``, …); SQLite
    is local and never network-hangs, so it is returned unchanged. A
    user-supplied ``connect_timeout`` is preserved (never overwritten), so an
    operator who deliberately set a longer bound keeps it. ``connect_timeout``
    is not a secret, so this respects the password-in-``.pgpass`` rule.

    Args:
        storage_url: The run's resolved Optuna storage URL.

    Returns:
        The URL with ``connect_timeout`` ensured for a postgres backend, or the
        original URL unchanged for any other backend (or an unparseable URL).
    """
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    parsed = urlsplit(storage_url)
    backend = parsed.scheme.split("+", 1)[0]
    if backend not in _CONNECT_TIMEOUT_BACKENDS:
        return storage_url
    query = parse_qsl(parsed.query, keep_blank_values=True)
    if any(key == "connect_timeout" for key, _value in query):
        return storage_url

    query.append(("connect_timeout", str(int(_LIVE_CONNECT_TIMEOUT_S))))
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
    )


def _open_live_study(root: "TuneRunRoot") -> "_ReadableStore":
    """Open the live ``OptunaStudyStore`` for ``root`` (called on a worker thread).

    Imports optuna lazily HERE (never at module import). The constructor opens
    the RDB study eagerly, so the storage URL is first passed through
    :func:`_ensure_connect_timeout` to bound the connect at the source.
    """
    from pathlib import Path
    from urllib.parse import urlsplit

    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    assert root.storage_url is not None  # guarded by the caller
    url = urlsplit(root.storage_url)
    database = url.path if url.scheme.split("+", 1)[0] == "sqlite" else ""
    if (
        database
        and database != "/:memory:"
        and not Path(database).exists()
    ):
        raise FileNotFoundError(database)
    return OptunaStudyStore(
        storage_url=_ensure_connect_timeout(root.storage_url),
        study_name=root.study_name,
        directions=root.directions,
        create=False,
    )


@dataclass(frozen=True)
class _StudySnapshot:
    """Detached live-study data used by one Monitor render."""

    trials: list["Trial"]
    _best: "Optional[Trial]"
    _pareto_front: list["Trial"]
    _importances: Optional[dict[str, float]]

    def best(self) -> "Optional[Trial]":
        """Return the captured winning trial."""
        return self._best

    def pareto_front(self) -> list["Trial"]:
        """Return the captured Pareto front."""
        return list(self._pareto_front)

    def param_importances(self) -> Optional[dict[str, float]]:
        """Return the last completed fANOVA model, when available."""
        return None if self._importances is None else dict(self._importances)


class _ImportanceCache:
    """Cache fANOVA by run and terminal count with one global refresh pending."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: Optional[tuple[str, str]] = None
        self._value: Optional[dict[str, float]] = None
        self._fresh_for: Optional[int] = None
        self._pending: "Optional[Future[Optional[dict[str, float]]]]" = None
        self._pending_key: Optional[tuple[str, str]] = None
        self._pending_for: Optional[int] = None

    def read(
        self,
        key: tuple[str, str],
        n_trials: int,
        refresh: "Callable[[], Optional[dict[str, float]]]",
    ) -> Optional[dict[str, float]]:
        """Return cached importances and schedule one refresh when stale."""
        with self._lock:
            if key != self._key:
                self._key = key
                self._value = None
                self._fresh_for = None
            pending = self._pending
            if pending is not None and pending.done():
                refreshed = _collect_importance_refresh(pending)
                if self._pending_key == key:
                    self._value = refreshed
                    self._fresh_for = self._pending_for
                self._pending = None
                self._pending_key = None
                self._pending_for = None
            if self._pending is None and self._fresh_for != n_trials:
                self._pending = _LIVE_IMPORTANCE_POOL.submit(refresh)
                self._pending_key = key
                self._pending_for = n_trials
            return None if self._value is None else dict(self._value)

    def wait_for_refresh(self, timeout: float) -> bool:
        """Wait for a pending refresh, for deterministic tests."""
        with self._lock:
            pending = self._pending
        if pending is None:
            return False
        try:
            pending.result(timeout=timeout)
        except FutureTimeout:
            return False
        except Exception:  # noqa: BLE001 - a failed refresh still finished
            return True
        return True

    def clear(self) -> None:
        """Forget cached and pending state, for test isolation."""
        with self._lock:
            self._key = None
            self._value = None
            self._fresh_for = None
            if self._pending is None or self._pending.done():
                self._pending = None
                self._pending_key = None
                self._pending_for = None


def _collect_importance_refresh(
    pending: "Future[Optional[dict[str, float]]]",
) -> Optional[dict[str, float]]:
    """Collect a completed fANOVA refresh without failing the Monitor."""
    try:
        return pending.result()
    except Exception:  # noqa: BLE001 - the figure degrades; the read does not
        logger.warning("param_importances refresh failed", exc_info=True)
        return None


_IMPORTANCES: _ImportanceCache = _ImportanceCache()


def _read_importances(store: "_ReadableStore") -> Optional[dict[str, float]]:
    """Read optional native importances without failing the Monitor."""
    getter = getattr(store, "param_importances", None)
    if getter is None:
        return None
    try:
        return dict(getter() or {})
    except Exception:  # noqa: BLE001 - the figure degrades; the read does not
        logger.warning("param_importances read failed", exc_info=True)
        return None


def _snapshot_live_study(root: "TuneRunRoot") -> _StudySnapshot:
    """Materialize every storage-backed field inside the poll deadline."""
    store = _open_live_study(root)
    trials = list(store.trials)
    terminal_trials = getattr(store, "terminal_trials", None)
    terminal_count = (
        len(terminal_trials()) if callable(terminal_trials) else len(trials)
    )
    return _StudySnapshot(
        trials=trials,
        _best=store.best(),
        _pareto_front=list(store.pareto_front()),
        _importances=_IMPORTANCES.read(
            (root.storage_url or "", root.study_name),
            terminal_count,
            lambda: _read_importances(store),
        ),
    )


def read_study_for_monitor(
    root: "TuneRunRoot",
) -> "tuple[Optional[_ReadableStore], str]":
    """Read the run's study for the Monitor view, degrading gracefully (OQ4).

    Resolution order:

    1. **Parquet-only run** (no ``storage_url``): read the finished
       ``trials.parquet`` directly — no live attempt, no note.
    2. **Live run, ``tune`` extra missing**: skip the live read, fall back to
       the journal, and return the "install the tune extra" note.
    3. **Live run, extra present**: open and snapshot the live study on a
       worker thread inside :data:`_LIVE_READ_TIMEOUT_S`. On success → detached
       data, no note. On timeout / read error → the journal plus a status note.

    The store is read-only and the poll must never raise, so every failure path
    degrades to the journal (or ``None`` when no journal exists yet).

    Args:
        root: The bound tune output handle.

    Returns:
        ``(store, note)``: the resolved store (or ``None`` when nothing is
        readable yet) and a degrade / status note (``""`` when the read was
        clean).
    """
    # 1. Parquet-only run — no live study to reach.
    if root.storage_url is None:
        return _load_journal(root), ""

    # 2. Live run, but the tune extra (optuna) is not installed.
    if importlib.util.find_spec("optuna") is None:
        return _load_journal(root), _NOTE_MISSING_EXTRA

    # 3. Live run, extra present — open and read with a bounded, non-re-blocking
    #    wait so unavailable storage cannot stall the poll. The wait here NEVER
    #    re-joins a still-working worker: we submit to the shared pool and,
    #    on timeout, return the parquet fallback immediately, leaving the
    #    orphaned future to finish (and be discarded) on its own. Using a
    #    ``with``-managed pool here would re-introduce the bug — its ``__exit__``
    #    calls ``shutdown(wait=True)``, re-joining the stuck worker and blocking
    #    the full connect duration regardless of the ``result`` timeout.
    future = _LIVE_READS.acquire(
        (root.storage_url, root.study_name),
        lambda: _LIVE_OPEN_POOL.submit(_snapshot_live_study, root),
    )
    if future is None:
        return _load_journal(root), _monitor_degrade_note(root, FutureTimeout())
    try:
        snapshot = future.result(timeout=_LIVE_READ_TIMEOUT_S)
        _LIVE_READS.consume(future)
        return snapshot, ""
    except FutureTimeout as exc:
        logger.warning(
            "Live tune study read timed out after %.1fs (url=%s); "
            "degrading to journal.",
            _LIVE_READ_TIMEOUT_S,
            root.storage_url,
        )
        return _load_journal(root), _monitor_degrade_note(root, exc)
    except Exception as exc:  # noqa: BLE001 - any open/read error degrades
        _LIVE_READS.consume(future)
        logger.warning(
            "Live tune study read failed (url=%s); degrading to journal.",
            root.storage_url,
            exc_info=True,
        )
        return _load_journal(root), _monitor_degrade_note(root, exc)


def _monitor_degrade_note(root: "TuneRunRoot", error: BaseException) -> str:
    """Pick the Monitor degrade note for a failed live-study open.

    A pre-cutover run (its marker/spec still names the legacy ``"tune"`` study)
    gets a friendly re-run message; everything else keeps the generic
    "couldn't reach the live study" note. Reuses the Phase-2 legacy-study
    detector (the same predicate the CLI startup guard uses) so the GUI and CLI
    can't disagree about what "legacy" means.

    Args:
        root: The bound tune output handle being read.
        error: The open/connect exception that triggered the degrade. Retained
            for intent + future detector hooks; the dispositive signal is the
            recorded ``study_name``, so a legacy run gets the legacy note even
            when the failure mode is a timeout.

    Returns:
        The friendly legacy note for a pre-cutover run, else the generic
        unreachable note.
    """
    # Lazy import: the helper lives behind the tune extra; importing it
    # function-local keeps _callbacks importable without optuna.
    from phenotypic.tune.strategy._optuna_support import is_legacy_study_name

    if is_legacy_study_name(root.study_name):
        return _NOTE_LEGACY_STUDY
    return _NOTE_LIVE_UNREACHABLE


# ---------------------------------------------------------------------------
# Monitor — render helpers
# ---------------------------------------------------------------------------

#: The trials-table columns, in display order.
_TRIALS_COLUMNS: tuple[str, ...] = ("number", "score", "n_images", "failed")


def _build_trials_table(store: "Optional[_ReadableStore]"):  # type: ignore[no-untyped-def]
    """Render the trials ``DataTable`` from ``store`` (a placeholder when empty)."""
    from dash import dash_table, html

    from phenotypic.gui._design import (
        COLOR_MUTED,
        COLOR_NAVY,
        FONT_FAMILY_MONO,
        FONT_SIZE_BODY_SM,
        FONT_SIZE_CAPTION,
    )

    if store is None or not store.trials:
        return html.P("No trials yet.", className="tune-monitor-note")

    rows = [
        {
            "number": t.number,
            "score": round(t.score, 4),
            "n_images": t.n_images,
            "failed": "yes" if t.failed else "",
        }
        for t in store.trials
    ]
    # Numeric columns render in mono per DESIGN.md "05 -- Data Tables": header
    # is 11px mono uppercase muted with a 2px navy underline; cells are mono.
    return dash_table.DataTable(  # type: ignore[attr-defined]
        data=rows,
        columns=[{"name": col, "id": col} for col in _TRIALS_COLUMNS],
        page_size=10,
        sort_action="native",
        style_cell={
            "fontFamily": FONT_FAMILY_MONO,
            "fontSize": FONT_SIZE_BODY_SM,
            "padding": "4px 8px",
            "textAlign": "left",
        },
        style_header={
            "fontFamily": FONT_FAMILY_MONO,
            "fontSize": FONT_SIZE_CAPTION,
            "fontWeight": "500",
            "textTransform": "uppercase",
            "letterSpacing": "0.08em",
            "color": COLOR_MUTED,
            "borderBottom": f"2px solid {COLOR_NAVY}",
        },
    )


def _gap_badge_outputs(store: "Optional[_ReadableStore]") -> tuple[str, str]:
    """The gap-badge ``(label, className)`` for ``store`` (empty → stable)."""
    from phenotypic.gui.tune._study_read import gap_badge

    if store is None:
        return "--", "tune-gap-badge tune-gap-badge-stable"
    label, flagged = gap_badge(store)
    variant = "unstable" if flagged else "stable"
    return label, f"tune-gap-badge tune-gap-badge-{variant}"


def _register_setup_picker_callbacks(app, sandbox) -> None:  # type: ignore[no-untyped-def]
    """Register both sandbox-bounded Setup file pickers."""
    from dash import ALL, Input, Output, no_update

    from phenotypic.gui.builder._directory_browser import directory_tree

    picker_specs = (
        (
            "pipeline",
            ids.TUNE_SETUP_PICK_PIPELINE,
            ids.TUNE_SETUP_PIPELINE_CANCEL,
            ids.TUNE_SETUP_PIPELINE_MODAL,
            ids.TUNE_SETUP_PIPELINE_BROWSE_DIR,
            ids.TUNE_SETUP_PIPELINE_MODAL_BODY,
            ids.TUNE_SETUP_PIPELINE_ENTRY,
            ids.TUNE_SETUP_PIPELINE_PICKER_STORE,
            PIPELINE_CONFIG_SUFFIXES | frozenset({CONFIG_SUFFIX_TUNING}),
        ),
        (
            "metadata",
            ids.TUNE_SETUP_PICK_METADATA,
            ids.TUNE_SETUP_METADATA_CANCEL,
            ids.TUNE_SETUP_METADATA_MODAL,
            ids.TUNE_SETUP_METADATA_BROWSE_DIR,
            ids.TUNE_SETUP_METADATA_MODAL_BODY,
            ids.TUNE_SETUP_METADATA_ENTRY,
            ids.TUNE_SETUP_METADATA_PICKER_STORE,
            frozenset({".csv", ".parquet"}),
        ),
    )

    for (
        kind,
        open_id,
        cancel_id,
        modal_id,
        browse_id,
        body_id,
        entry_type,
        store_id,
        extensions,
    ) in picker_specs:

        @app.callback(
            Output(modal_id, "is_open", allow_duplicate=True),
            Input(open_id, "n_clicks"),
            Input(cancel_id, "n_clicks"),
            prevent_initial_call=True,
        )
        def _toggle_picker(open_clicks, cancel_clicks, _open=open_id, _cancel=cancel_id):  # type: ignore[no-untyped-def]
            if ctx.triggered_id == _open and open_clicks:
                return True
            if ctx.triggered_id == _cancel and cancel_clicks:
                return False
            return no_update

        @app.callback(
            Output(body_id, "children"),
            Input(browse_id, "data"),
            prevent_initial_call=True,
        )
        def _render_picker_body(
            current,
            _extensions=extensions,
            _entry_type=entry_type,
        ):  # type: ignore[no-untyped-def]
            return directory_tree(
                sandbox.root,
                current=Path(current) if current else None,
                extensions=_extensions,
                select_files=True,
                id_type=_entry_type,
            )

        @app.callback(
            Output(browse_id, "data", allow_duplicate=True),
            Output(store_id, "data"),
            Output(modal_id, "is_open", allow_duplicate=True),
            Input(
                {"type": entry_type, "kind": ALL, "path": ALL},
                "n_clicks",
            ),
            prevent_initial_call=True,
        )
        def _select_picker_entry(
            _clicks,
            _entry_type=entry_type,
            _kind=kind,
        ):  # type: ignore[no-untyped-def]
            triggered = ctx.triggered_id
            if (
                not isinstance(triggered, dict)
                or triggered.get("type") != _entry_type
                or not ctx.triggered
                or not ctx.triggered[0].get("value")
            ):
                return no_update, no_update, no_update
            selected = triggered.get("path")
            entry_kind = triggered.get("kind")
            if not isinstance(selected, str):
                return no_update, no_update, no_update
            if entry_kind in {"dir", "parent"}:
                return selected, no_update, no_update
            payload = setup_path_payload(
                sandbox,
                selected,
                kind=_kind,
            )
            if payload is None:
                return no_update, no_update, no_update
            # Always emit a fresh payload, even for the same path, so an
            # explicit re-selection repairs stale or mismatched session state.
            return no_update, payload, False


def register_callbacks(app, *, sandbox=None) -> None:  # type: ignore[no-untyped-def]
    """Register the tune sub-app's Dash callbacks on ``app``.

    Wires:

    * **Sub-tab switch** — a click on any of the four sub-tab buttons re-resolves
      the active view via :func:`active_view`, toggles each view container's
      visibility and each button's active class, and mirrors the active view
      name into :data:`ids.TUNE_ACTIVE_VIEW_STORE`.
    * **Monitor poll** — every 3 s the poll re-reads the study
      (:func:`read_study_for_monitor`) and re-renders the objective / importance
      figures, the gap badge, the trials table, and the degrade note.
    * **Curate** (when ``sandbox`` is bound) — the sandbox-bounded Image Source
      picker, the A/B pin + overlay render/poll, the mode toggle, and the
      "Set as winner" write (see :func:`_register_curate_callbacks`).

    Args:
        app: The :class:`dash.Dash` instance whose layout is assigned.
        sandbox: The frozen-at-launch sandbox root. When ``None`` the Curate
            callbacks are skipped (the Curate view degrades to a note).
    """
    from dash import ALL, Input, Output, State, html

    setup_drafts = SetupDraftCache()

    @app.callback(
        Output(ids.TUNE_ACTIVE_DESTINATION_STORE, "data"),
        *[
            Output(_nav.destination_view_id(name), "className")
            for name in _nav.DESTINATIONS
        ],
        *[
            Output(_nav.destination_button_id(name), "className")
            for name in _nav.DESTINATIONS
        ],
        *[
            Input(_nav.destination_button_id(name), "n_clicks")
            for name in _nav.DESTINATIONS
        ],
        State(ids.TUNE_SETUP_AUTHORED_SPEC_STORE, "data"),
        State(ids.TUNE_SETUP_DRAFT_STORE, "data"),
        prevent_initial_call=True,
    )
    def _switch_destination(*args: object) -> tuple[str, ...]:
        descriptor = args[-2] if len(args) >= 2 else None
        draft = setup_draft_from_store(
            args[-1] if args else None,
            cache=setup_drafts,
        )
        spec_path = active_authored_spec_for_draft(
            descriptor,
            draft,
        )
        active = _nav.active_destination(
            ctx.triggered_id,
            pipeline_path=spec_path,
        )
        view_classes = [
            _nav.destination_view_class(name, active) for name in _nav.DESTINATIONS
        ]
        button_classes = [
            _nav.destination_button_class(name, active) for name in _nav.DESTINATIONS
        ]
        return (active, *view_classes, *button_classes)

    if sandbox is not None:
        _register_setup_picker_callbacks(app, sandbox)

    @app.callback(
        Output(ids.TUNE_SETUP_PIPELINE_STORE, "data"),
        Output(ids.TUNE_SETUP_METADATA_STORE, "data"),
        Output(ids.TUNE_SETUP_PIPELINE_SOURCE, "children"),
        Output(ids.TUNE_SETUP_METADATA_SOURCE, "children"),
        Input(ids.TUNE_SETUP_PIPELINE_INPUT, "value"),
        Input(ids.TUNE_SETUP_PIPELINE_PICKER_STORE, "data"),
        Input(TUNE_PIPELINE_PATH_STORE, "data", allow_optional=True),
        Input(ids.TUNE_SETUP_METADATA_INPUT, "value"),
        Input(ids.TUNE_SETUP_METADATA_PICKER_STORE, "data"),
        Input(SHELL_METADATA_CSV_STORE, "data", allow_optional=True),
    )
    def _resolve_setup_inputs(
        typed_pipeline: str | None,
        pipeline_picker: object,
        pipeline_handoff: object,
        typed_metadata: str | None,
        metadata_picker: object,
        metadata_handoff: object,
    ) -> tuple[object, ...]:
        if sandbox is None:
            return (
                None,
                None,
                "Pipeline: unavailable without sandbox",
                "Metadata: unavailable without sandbox",
            )
        pipeline = resolve_setup_path(
            sandbox=sandbox,
            kind="pipeline",
            typed_path=typed_pipeline,
            picker_payload=pipeline_picker,
            shared_payload=pipeline_handoff,
        )
        metadata = resolve_setup_path(
            sandbox=sandbox,
            kind="metadata",
            typed_path=typed_metadata,
            picker_payload=metadata_picker,
            shared_payload=metadata_handoff,
        )
        return (
            pipeline.to_store(),
            metadata.to_store(),
            _source_label("Pipeline", pipeline),
            _source_label("Metadata", metadata),
        )

    @app.callback(
        Output(ids.TUNE_SETUP_DRAFT_STORE, "data"),
        Input(ids.TUNE_SETUP_PIPELINE_STORE, "data"),
        Input(ids.TUNE_SETUP_METADATA_STORE, "data"),
        Input(ids.TUNE_SETUP_REPLACE_SCORER, "value"),
        Input({"type": ids.TUNE_SETUP_SPACE_TUNABLE, "key": ALL}, "id"),
        Input({"type": ids.TUNE_SETUP_SPACE_LOW, "key": ALL}, "id"),
        Input({"type": ids.TUNE_SETUP_SPACE_LOW, "key": ALL}, "value"),
        Input({"type": ids.TUNE_SETUP_SPACE_HIGH, "key": ALL}, "id"),
        Input({"type": ids.TUNE_SETUP_SPACE_HIGH, "key": ALL}, "value"),
        Input({"type": ids.TUNE_SETUP_SPACE_LOG, "key": ALL}, "id"),
        Input({"type": ids.TUNE_SETUP_SPACE_LOG, "key": ALL}, "value"),
        Input({"type": ids.TUNE_SETUP_SPACE_CHOICES, "key": ALL}, "id"),
        Input({"type": ids.TUNE_SETUP_SPACE_CHOICES, "key": ALL}, "value"),
        Input({"type": ids.TUNE_SETUP_SPACE_TUNABLE, "key": ALL}, "value"),
    )
    def _update_setup_draft(
        pipeline_store: object,
        metadata_store: object,
        replace_values: object,
        tunable_ids: list[dict[str, object]],
        low_ids: list[dict[str, object]],
        lows: list[object],
        high_ids: list[dict[str, object]],
        highs: list[object],
        log_ids: list[dict[str, object]],
        logs: list[object],
        choice_ids: list[dict[str, object]],
        choices: list[object],
        tunables: list[object],
    ) -> dict[str, object]:
        keys = [
            entry.get("key") if isinstance(entry, dict) else None
            for entry in (tunable_ids or [])
        ]
        edits = _collect_space_edits(
            keys,
            lows or [],
            highs or [],
            logs or [],
            choices or [],
            tunables or [],
            low_keys=low_ids,
            high_keys=high_ids,
            log_keys=log_ids,
            choice_keys=choice_ids,
        )
        draft = build_setup_draft(
            pipeline=setup_path_resolution_from_store(
                pipeline_store,
                sandbox=sandbox,
                kind="pipeline",
            ),
            metadata=setup_path_resolution_from_store(
                metadata_store,
                sandbox=sandbox,
                kind="metadata",
            ),
            edits=edits,
            replace_scorer=_toggle_on(replace_values),
        )
        return setup_drafts.publish(draft)

    @app.callback(
        Output(ids.TUNE_SETUP_SEARCH_SPACE, "children"),
        Output(ids.TUNE_SETUP_SEARCH_SPACE, "className"),
        Output(ids.TUNE_SETUP_SCORER_NOTE, "children"),
        Output(ids.TUNE_SETUP_SCORER, "className"),
        Output(ids.TUNE_SETUP_REPLACE_SCORER, "options"),
        Output(ids.TUNE_SETUP_RENDER_REVISION_STORE, "data"),
        Input(ids.TUNE_SETUP_DRAFT_STORE, "data"),
        State(ids.TUNE_SETUP_RENDER_REVISION_STORE, "data"),
    )
    def _render_setup_draft(
        draft_store: object,
        rendered_source_revision: object,
    ) -> tuple[object, object, str, str, list[dict[str, object]], object]:
        draft = setup_draft_from_store(draft_store, cache=setup_drafts)
        locked = "tune-setup-section tune-setup-locked"
        replace_options = [
            {
                "label": "Replace scorer with metadata-backed QC scorer",
                "value": "on",
                "disabled": not bool(draft and draft.source_is_spec),
            }
        ]
        if draft is None or draft.pipeline_path is None:
            issue = (
                "\n".join(draft.issues)
                if draft is not None and draft.issues
                else "Choose a pipeline or existing tuning spec."
            )
            return (
                [html.H3("Search space"), html.Div(issue)],
                locked,
                "Locked until the pipeline/spec validates.",
                locked,
                replace_options,
                draft.source_revision if draft is not None else None,
            )
        scorer_note = (
            f"Preserving existing {draft.scorer_name or 'scorer'}."
            if draft.source_is_spec and not draft.replace_scorer
            else "A metadata-backed QC scorer will be authored."
        )
        if rendered_source_revision == draft.source_revision:
            return (
                no_update,
                no_update,
                scorer_note,
                "tune-setup-section",
                replace_options,
                no_update,
            )
        try:
            from phenotypic.gui.tune._space import setup_knob_forms

            rows = setup_knob_forms(load_pipeline_or_spec(Path(draft.pipeline_path)))
        except Exception as exc:  # noqa: BLE001
            return (
                [html.H3("Search space"), html.Div(f"Could not load: {exc}")],
                locked,
                f"Could not load pipeline/spec: {exc}",
                locked,
                replace_options,
                draft.source_revision,
            )
        return (
            [
                html.H3("Search space"),
                html.Div(rows, className="tune-space-knobs"),
            ],
            "tune-setup-section" if rows else locked,
            scorer_note,
            "tune-setup-section" if rows else locked,
            replace_options,
            draft.source_revision,
        )

    @app.callback(
        Output(ids.TUNE_SETUP_CONTINUE, "disabled"),
        Output(ids.TUNE_SETUP_GATE, "children"),
        Output(_nav.destination_button_id("run"), "disabled"),
        Output(ids.TUNE_SETUP_SIGNATURE_STORE, "data"),
        Input(ids.TUNE_SETUP_DRAFT_STORE, "data"),
        Input(ids.TUNE_SETUP_AUTHORED_SPEC_STORE, "data"),
    )
    def _validate_setup(
        draft_store: object,
        descriptor: object,
    ) -> tuple[bool, object, bool, str]:
        draft = setup_draft_from_store(draft_store, cache=setup_drafts)
        if draft is None:
            return True, "Choose a pipeline or existing tuning spec.", True, ""
        authored = active_authored_spec_for_draft(descriptor, draft)
        if authored:
            note = f"Authored tuning spec: {authored}"
        elif draft.issues:
            note = "\n".join(draft.issues)
        else:
            note = "Setup is valid. Continue to author the tuning spec."
        return (
            not draft.is_valid,
            note,
            _nav.destination_button_disabled("run", pipeline_path=authored),
            draft.revision,
        )

    @app.callback(
        Output(ids.TUNE_SETUP_AUTHORED_SPEC_STORE, "data"),
        Output(ids.TUNE_SETUP_GATE, "children", allow_duplicate=True),
        Output(ids.TUNE_ACTIVE_DESTINATION_STORE, "data", allow_duplicate=True),
        *[
            Output(_nav.destination_view_id(name), "className", allow_duplicate=True)
            for name in _nav.DESTINATIONS
        ],
        *[
            Output(_nav.destination_button_id(name), "className", allow_duplicate=True)
            for name in _nav.DESTINATIONS
        ],
        Output(ids.TUNE_RUN_STRATEGY, "value"),
        Output(ids.TUNE_RUN_N_TRIALS, "value"),
        Output(ids.TUNE_RUN_STORAGE_MODE, "value"),
        Output(ids.TUNE_RUN_STORAGE_URL, "value"),
        Output(ids.TUNE_RUN_STORAGE_ENV, "value"),
        Input(ids.TUNE_SETUP_CONTINUE, "n_clicks"),
        State(ids.TUNE_SETUP_DRAFT_STORE, "data"),
        prevent_initial_call=True,
    )
    def _author_setup_spec(
        n_clicks: int | None,
        draft_store: object,
    ) -> tuple[object, ...]:
        if not n_clicks:
            return (no_update,) * 14
        if sandbox is None:
            return (
                "",
                "Setup authoring requires a sandbox-bound GUI launch.",
                *([no_update] * 12),
            )
        draft = setup_draft_from_store(draft_store, cache=setup_drafts)
        if draft is None or not draft.is_valid:
            return (
                "",
                "\n".join(draft.issues) if draft is not None else (
                    "Setup draft is invalid; review Setup and try again."
                ),
                *([no_update] * 12),
            )
        try:
            write_receipt = write_setup_draft_receipt(
                sandbox_root=sandbox.root,
                draft=draft,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tune setup authoring failed")
            return (
                "",
                f"Could not author tuning spec: {exc}",
                *([no_update] * 12),
            )
        authored = write_receipt.path
        launch_defaults = authored_spec_launch_defaults(authored)
        active: _nav.Destination = "run"
        view_classes = [
            _nav.destination_view_class(name, active) for name in _nav.DESTINATIONS
        ]
        button_classes = [
            _nav.destination_button_class(name, active) for name in _nav.DESTINATIONS
        ]
        return (
            authored_spec_descriptor(
                path=str(authored),
                pipeline_path=draft.pipeline_path or "",
                metadata_path=draft.metadata_path,
                replace_scorer=draft.replace_scorer,
                setup_signature=draft.revision,
                launch_defaults=launch_defaults,
                write_receipt=write_receipt,
            ),
            f"Authored tuning spec: {authored}",
            active,
            *view_classes,
            *button_classes,
            launch_defaults["strategy"],
            launch_defaults["n_trials"],
            "spec",
            "",
            DEFAULT_STORAGE_ENV,
        )

    @app.callback(
        Output(ids.TUNE_RUN_COMMAND, "children"),
        Output(ids.TUNE_RUN_PORTABLE_COMMAND, "children"),
        Output(ids.TUNE_RUN_PREFLIGHT, "children"),
        Output(ids.TUNE_RUN_DEPLOY, "disabled"),
        Output(ids.TUNE_RUN_COPY, "style"),
        Input(ids.TUNE_SETUP_AUTHORED_SPEC_STORE, "data"),
        Input(ids.TUNE_SETUP_DRAFT_STORE, "data"),
        Input(SHELL_SOURCE_IMAGE_ROOT_STORE, "data", allow_optional=True),
        Input(ids.TUNE_RUN_IMAGES_OVERRIDE, "value"),
        Input(ids.TUNE_RUN_OUTPUT_DIR, "value"),
        Input(ids.TUNE_RUN_STRATEGY, "value"),
        Input(ids.TUNE_RUN_N_TRIALS, "value"),
        Input(ids.TUNE_RUN_STORAGE_MODE, "value"),
        Input(ids.TUNE_RUN_STORAGE_URL, "value"),
        Input(ids.TUNE_RUN_STORAGE_ENV, "value"),
        Input(ids.TUNE_RUN_N_WORKERS, "value"),
        Input(ids.TUNE_RUN_SLURM_PARTITION, "value"),
        Input(ids.TUNE_RUN_SLURM_MEM, "value"),
        Input(ids.TUNE_RUN_SLURM_TIME, "value"),
        Input(ids.TUNE_RUN_HELD_OUT_FRACTION, "value"),
        Input(ids.TUNE_RUN_CV_GROUP, "value"),
        Input(ids.TUNE_RUN_MODE, "value"),
        Input(ids.TUNE_RUN_SCREEN, "value"),
    )
    def _render_run_command(
        authored_spec_descriptor_value: object,
        setup_draft_store: object,
        shared_source: object,
        images_override: str | None,
        output_dir: str | None,
        strategy: str | None,
        n_trials: object,
        storage_mode: str | None,
        storage_local_path: str | None,
        storage_environment_name: str | None,
        n_workers: object,
        slurm_partition: str | None,
        slurm_mem: str | None,
        slurm_time: str | None,
        held_out_fraction: object,
        cv_group: str | None,
        mode: str | None,
        screen_values: object,
    ) -> tuple[str, str, str, bool, dict[str, str]]:
        if sandbox is None:
            return (
                "",
                "",
                "Run requires a sandbox-bound GUI launch.",
                True,
                {"display": "none"},
            )
        command = _build_command_from_controls(
            sandbox=sandbox,
            authored_descriptor=authored_spec_descriptor_value,
            pipeline_store=None,
            metadata_store=None,
            replace_values=None,
            setup_signature=None,
            shared_source=shared_source,
            images_override=images_override,
            output_dir=output_dir,
            strategy=strategy,
            n_trials=n_trials,
            storage_mode=storage_mode,
            storage_local_path=storage_local_path,
            storage_environment_name=storage_environment_name,
            n_workers=n_workers,
            slurm_partition=slurm_partition,
            slurm_mem=slurm_mem,
            slurm_time=slurm_time,
            held_out_fraction=held_out_fraction,
            cv_group=cv_group,
            mode=mode,
            screen_values=screen_values,
            setup_draft_store=setup_draft_store,
            setup_draft_cache=setup_drafts,
        )
        note = "\n".join(command.issues) if command.issues else "Ready to deploy."
        return (
            command.display_command(),
            command.portable_command(),
            note,
            not command.deploy_eligible,
            {} if command.copy_eligible else {"display": "none"},
        )

    @app.callback(
        Output(ids.TUNE_RUN_STATUS, "children"),
        Output(ids.TUNE_RUN_ACTIVE_RECORD_STORE, "data"),
        Output(ids.TUNE_MONITOR_ACTIVE_RUN_STORE, "data", allow_duplicate=True),
        Output(ids.TUNE_ACTIVE_DESTINATION_STORE, "data", allow_duplicate=True),
        *[
            Output(_nav.destination_view_id(name), "className", allow_duplicate=True)
            for name in _nav.DESTINATIONS
        ],
        *[
            Output(_nav.destination_button_id(name), "className", allow_duplicate=True)
            for name in _nav.DESTINATIONS
        ],
        Input(ids.TUNE_RUN_DEPLOY, "n_clicks"),
        State(ids.TUNE_SETUP_AUTHORED_SPEC_STORE, "data"),
        State(ids.TUNE_SETUP_DRAFT_STORE, "data"),
        State(SHELL_SOURCE_IMAGE_ROOT_STORE, "data", allow_optional=True),
        State(ids.TUNE_RUN_IMAGES_OVERRIDE, "value"),
        State(ids.TUNE_RUN_OUTPUT_DIR, "value"),
        State(ids.TUNE_RUN_STRATEGY, "value"),
        State(ids.TUNE_RUN_N_TRIALS, "value"),
        State(ids.TUNE_RUN_STORAGE_MODE, "value"),
        State(ids.TUNE_RUN_STORAGE_URL, "value"),
        State(ids.TUNE_RUN_STORAGE_ENV, "value"),
        State(ids.TUNE_RUN_N_WORKERS, "value"),
        State(ids.TUNE_RUN_SLURM_PARTITION, "value"),
        State(ids.TUNE_RUN_SLURM_MEM, "value"),
        State(ids.TUNE_RUN_SLURM_TIME, "value"),
        State(ids.TUNE_RUN_HELD_OUT_FRACTION, "value"),
        State(ids.TUNE_RUN_CV_GROUP, "value"),
        State(ids.TUNE_RUN_MODE, "value"),
        State(ids.TUNE_RUN_SCREEN, "value"),
        prevent_initial_call=True,
    )
    def _deploy_run(
        n_clicks: int | None,
        authored_spec_descriptor_value: object,
        setup_draft_store: object,
        shared_source: object,
        images_override: str | None,
        output_dir: str | None,
        strategy: str | None,
        n_trials: object,
        storage_mode: str | None,
        storage_local_path: str | None,
        storage_environment_name: str | None,
        n_workers: object,
        slurm_partition: str | None,
        slurm_mem: str | None,
        slurm_time: str | None,
        held_out_fraction: object,
        cv_group: str | None,
        mode: str | None,
        screen_values: object,
    ) -> tuple[object, ...]:
        if not n_clicks:
            return (no_update,) * 10
        if sandbox is None:
            return ("Run requires a sandbox-bound GUI launch.", no_update, *([no_update] * 8))
        runner = app.server.config.get(CFG_RUNNER)
        registry = app.server.config.get(CFG_RUN_REGISTRY)
        if runner is None or registry is None:
            return ("Runner unavailable.", no_update, *([no_update] * 8))
        command = _build_command_from_controls(
            sandbox=sandbox,
            authored_descriptor=authored_spec_descriptor_value,
            pipeline_store=None,
            metadata_store=None,
            replace_values=None,
            setup_signature=None,
            shared_source=shared_source,
            images_override=images_override,
            output_dir=output_dir,
            strategy=strategy,
            n_trials=n_trials,
            storage_mode=storage_mode,
            storage_local_path=storage_local_path,
            storage_environment_name=storage_environment_name,
            n_workers=n_workers,
            slurm_partition=slurm_partition,
            slurm_mem=slurm_mem,
            slurm_time=slurm_time,
            held_out_fraction=held_out_fraction,
            cv_group=cv_group,
            mode=mode,
            screen_values=screen_values,
            setup_draft_store=setup_draft_store,
            setup_draft_cache=setup_drafts,
        )
        if not command.deploy_eligible or command.output_dir is None:
            return ("\n".join(command.issues), no_update, *([no_update] * 8))
        try:
            receipt = deploy_tune_run(
                runner=runner,
                registry=registry,
                sandbox=sandbox,
                argv=list(command.argv),
                output_dir=command.output_dir,
                slurm=command.execution_target == "slurm",
            )
        except Exception:  # noqa: BLE001
            logger.exception("Tune deploy failed")
            return (
                "Deploy failed. See the server log for details.",
                no_update,
                *([no_update] * 8),
            )
        run_id = receipt["run_id"]
        active: _nav.Destination = "monitor"
        view_classes = [
            _nav.destination_view_class(name, active) for name in _nav.DESTINATIONS
        ]
        button_classes = [
            _nav.destination_button_class(name, active) for name in _nav.DESTINATIONS
        ]
        return (
            f"Deployed: {run_id}",
            {**receipt, "mode": command.execution_target},
            receipt,
            active,
            *view_classes,
            *button_classes,
        )

    @app.callback(
        Output(ids.TUNE_MONITOR_SWITCHER, "children"),
        Output(ids.TUNE_MONITOR_CANCEL, "disabled"),
        Output(ids.TUNE_MONITOR_LOCAL_LOG, "children"),
        Output(ids.TUNE_MONITOR_SLURM_FLEET, "children"),
        Input(ids.TUNE_STUDY_POLL, "n_intervals"),
        Input(ids.TUNE_MONITOR_ACTIVE_RUN_STORE, "data"),
    )
    def _render_monitor_registry(
        _n: int | None,
        active_receipt: object,
    ) -> tuple[object, bool, str, str]:
        registry = app.server.config.get(CFG_RUN_REGISTRY)
        if registry is None:
            return "No run registry.", True, "", ""
        runner = app.server.config.get(CFG_RUNNER)
        if runner is not None and parse_run_receipt(active_receipt) is not None:
            reconcile_run_status(
                runner=runner,
                registry=registry,
                receipt=active_receipt,
            )
        records = registry.list()
        items = run_switcher_items(
            records,
            active_receipt=active_receipt,
        )
        active_item = next((item for item in items if item.active), None)
        switcher = [
            html.Button(
                f"{item.run_id} | {item.mode} | {item.status}",
                id={
                    "type": ids.TUNE_MONITOR_RUN_SWITCH,
                    "run_id": item.run_id,
                    "generation": item.generation or "",
                },
                n_clicks=0,
                className=(
                    "tune-monitor-switcher-item"
                    + (" tune-monitor-switcher-active" if item.active else "")
                ),
            )
            for item in items
        ]
        cancel_disabled = active_item is None or not active_item.killable
        local_text = ""
        slurm_text = ""
        if active_item is not None and active_item.mode == "local":
            lines = []
            identity = parse_run_receipt(active_receipt)
            if runner is not None and identity is not None:
                lines = runner.snapshot_log(
                    identity[0],
                    generation=identity[1],
                    tail=40,
                )
            local_text = "\n".join(lines) if lines else "No local log lines yet."
        elif active_item is not None:
            slurm_text = (
                f"SLURM run {active_item.run_id}: {active_item.status}. "
                "Cancellation is not supported in v1."
            )
        return switcher, cancel_disabled, local_text, slurm_text

    @app.callback(
        Output(ids.TUNE_MONITOR_ACTIVE_RUN_STORE, "data", allow_duplicate=True),
        Input(
            {
                "type": ids.TUNE_MONITOR_RUN_SWITCH,
                "run_id": ALL,
                "generation": ALL,
            },
            "n_clicks",
        ),
        prevent_initial_call=True,
    )
    def _select_monitor_run(_clicks: object) -> object:
        triggered = ctx.triggered_id
        if isinstance(triggered, dict) and triggered.get("type") == ids.TUNE_MONITOR_RUN_SWITCH:
            run_id = triggered.get("run_id")
            generation = triggered.get("generation")
            receipt = {"run_id": run_id, "generation": generation}
            return (
                receipt
                if parse_run_receipt(receipt) is not None
                else no_update
            )
        return no_update

    @app.callback(
        Output(ids.TUNE_MONITOR_CANCEL_NOTE, "children"),
        Input(ids.TUNE_MONITOR_CANCEL_CONFIRM, "submit_n_clicks"),
        State(ids.TUNE_MONITOR_ACTIVE_RUN_STORE, "data"),
        prevent_initial_call=True,
    )
    def _cancel_monitor_run(
        n_clicks: int | None,
        active_receipt: object,
    ) -> str:
        if not n_clicks:
            return ""
        runner = app.server.config.get(CFG_RUNNER)
        registry = app.server.config.get(CFG_RUN_REGISTRY)
        if runner is None or registry is None:
            return "Runner unavailable."
        return cancel_monitor_run(
            runner=runner,
            registry=registry,
            receipt=active_receipt,
        )

    @app.callback(
        Output(ids.TUNE_MONITOR_EXPORT_NOTE, "children"),
        Input(ids.TUNE_MONITOR_EXPORT, "n_clicks"),
        State(ids.TUNE_MONITOR_ACTIVE_RUN_STORE, "data"),
        prevent_initial_call=True,
    )
    def _export_monitor_best(
        n_clicks: int | None,
        active_receipt: object,
    ) -> str:
        if not n_clicks:
            return ""
        registry = app.server.config.get(CFG_RUN_REGISTRY)
        if registry is None:
            return "Run registry unavailable."
        try:
            written = export_monitor_best_pipeline(
                registry=registry,
                receipt=active_receipt,
            )
        except FileNotFoundError as exc:
            return f"Export unavailable: {exc}"
        except Exception:  # noqa: BLE001
            logger.warning("Monitor best-pipeline export failed", exc_info=True)
            return "Export failed -- see the server log."
        return f"Exported {written}"

    @app.callback(
        Output(ids.TUNE_ACTIVE_VIEW_STORE, "data"),
        *[
            Output(ids.view_container_id(name), "className")
            for name in ids.SUBTAB_ORDER
        ],
        *[
            Output(ids.subtab_button_id(name), "className")
            for name in ids.SUBTAB_ORDER
        ],
        *[
            Input(ids.subtab_button_id(name), "n_clicks")
            for name in ids.SUBTAB_ORDER
        ],
        prevent_initial_call=True,
    )
    def _switch_subtab(*_n_clicks: int) -> tuple[str, ...]:
        active = active_view(ctx.triggered_id)
        container_classes = [
            ids.view_container_class(name, active) for name in ids.SUBTAB_ORDER
        ]
        button_classes = [
            ids.subtab_button_class(name, active) for name in ids.SUBTAB_ORDER
        ]
        return (active, *container_classes, *button_classes)

    @app.callback(
        Output(ids.TUNE_OBJECTIVE_FIGURE, "figure"),
        Output(ids.TUNE_IMPORTANCE_FIGURE, "figure"),
        Output(ids.TUNE_GAP_BADGE, "children"),
        Output(ids.TUNE_GAP_BADGE, "className"),
        Output(ids.TUNE_TRIALS_TABLE, "children"),
        Output(ids.TUNE_MONITOR_NOTE, "children"),
        Input(ids.TUNE_STUDY_POLL, "n_intervals"),
        State(ids.TUNE_RUN_ROOT_STORE, "data"),
    )
    def _poll_study(_n_intervals: int, run_root_data: "Optional[dict]"):  # type: ignore[no-untyped-def]
        from phenotypic.gui.tune._run_root import TuneRunRoot
        from phenotypic.gui.tune._study_read import (
            build_importance_figure,
            build_objective_figure,
        )

        # Re-discover the bound run from its path each tick (cheap: reads the
        # markers, never optuna). A missing/invalid store renders empty figures.
        store: "Optional[_ReadableStore]" = None
        note = ""
        if run_root_data and run_root_data.get("path"):
            from pathlib import Path

            try:
                root = TuneRunRoot.discover(Path(run_root_data["path"]))
                store, note = read_study_for_monitor(root)
            except Exception:  # noqa: BLE001 - poll must never raise
                logger.warning("Monitor poll re-discovery failed", exc_info=True)

        trials = list(store.trials) if store is not None else []
        objective_fig = build_objective_figure(trials)

        importances = _param_importances(store)
        importance_fig = build_importance_figure(importances)

        badge_label, badge_class = _gap_badge_outputs(store)
        table = _build_trials_table(store)
        return objective_fig, importance_fig, badge_label, badge_class, table, note

    # The Launch command mirror and the Space export are sandbox-independent
    # (Launch only renders a string; Space only reads a pipeline and writes
    # tuning_spec.json — neither re-optimizes) so both register unconditionally.
    _register_launch_command_mirror(app)
    _register_space_export(app)

    if sandbox is not None:
        # The runtime run picker + the Curate Image Source picker both need the
        # sandbox boundary. Registered unconditionally (whenever a sandbox is
        # bound) so they are wired even for the empty-state mount — the run
        # picker is exactly what populates ``TUNE_RUN_ROOT_STORE`` and swaps in
        # the loaded views, and ``suppress_callback_exceptions=True`` makes
        # registering against not-yet-present loaded-view components safe.
        _register_run_picker_callbacks(app, sandbox)
        _register_curate_callbacks(app, sandbox)


# ---------------------------------------------------------------------------
# Run picker — bind a tune output directory at runtime (Chunk C)
# ---------------------------------------------------------------------------

def _register_run_picker_callbacks(app, sandbox) -> None:  # type: ignore[no-untyped-def]
    """Register the runtime run-picker callbacks (Chunk C).

    Wires the sandbox-bounded run-directory picker that turns the empty-state
    mount into a loaded co-pilot:

    * **Open / cancel** the picker modal.
    * **Navigate** the folder-only tree (folder click → browse-dir store).
    * **Re-render** the tree body on browse-dir change.
    * **Confirm → bind**: validate the chosen directory via
      :func:`~phenotypic.gui.tune._run_picker.discover_run_payload`, and on
      success write the run-root payload into ``TUNE_RUN_ROOT_STORE`` AND swap
      the page body to the loaded four-view layout (the Monitor / Curate / Space
      / Launch callbacks then render from the store). On failure the store /
      body are left untouched and a clear note is shown — never a 500.

    Binding only **reads** the run directory; it never mutates it.

    Args:
        app: The :class:`dash.Dash` instance.
        sandbox: The frozen-at-launch sandbox bounding the directory selection
            (and threaded into the loaded body's Curate Image Source picker).
    """
    from dash import ALL, Input, Output, State, no_update

    from phenotypic.gui.tune._layout import build_loaded_body
    from phenotypic.gui.tune._run_picker import (
        discover_run_payload,
        render_run_picker_tree,
    )
    from phenotypic.gui.tune._run_root import TuneRunRoot

    # --- Open / cancel the picker modal -----------------------------------
    @app.callback(
        Output(ids.TUNE_RUN_PICKER_MODAL, "is_open", allow_duplicate=True),
        Input(ids.TUNE_BTN_PICK_RUN, "n_clicks"),
        Input(ids.TUNE_BTN_RUN_PICKER_CANCEL, "n_clicks"),
        prevent_initial_call=True,
    )
    def _toggle_run_picker_modal(open_clicks, cancel_clicks):  # type: ignore[no-untyped-def]
        if ctx.triggered_id == ids.TUNE_BTN_PICK_RUN and open_clicks:
            return True
        if ctx.triggered_id == ids.TUNE_BTN_RUN_PICKER_CANCEL and cancel_clicks:
            return False
        return no_update

    # --- Navigate the tree (folder click → browse-dir store) --------------
    @app.callback(
        Output(ids.TUNE_RUN_PICKER_BROWSE_DIR, "data", allow_duplicate=True),
        Input(
            {"type": ids.TUNE_DIR_ENTRY_RUN, "kind": ALL, "path": ALL},
            "n_clicks",
        ),
        prevent_initial_call=True,
    )
    def _navigate_run_picker_tree(_clicks):  # type: ignore[no-untyped-def]
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return no_update
        if triggered.get("type") != ids.TUNE_DIR_ENTRY_RUN:
            return no_update
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return no_update
        path = triggered.get("path")
        return path if isinstance(path, str) else no_update

    # --- Re-render the tree body on browse-dir change ---------------------
    @app.callback(
        Output(ids.TUNE_RUN_PICKER_MODAL_BODY, "children"),
        Input(ids.TUNE_RUN_PICKER_BROWSE_DIR, "data"),
        prevent_initial_call=True,
    )
    def _render_run_picker_body(dir_value):  # type: ignore[no-untyped-def]
        from pathlib import Path

        current = Path(dir_value) if dir_value else None
        return render_run_picker_tree(sandbox, current)

    # --- Confirm → discover + bind the run --------------------------------
    @app.callback(
        Output(ids.TUNE_RUN_ROOT_STORE, "data", allow_duplicate=True),
        Output(ids.TUNE_PAGE_BODY, "children", allow_duplicate=True),
        Output(ids.TUNE_RUN_PICKER_LABEL, "children", allow_duplicate=True),
        Output(ids.TUNE_RUN_PICKER_NOTE, "children", allow_duplicate=True),
        Output(ids.TUNE_RUN_PICKER_MODAL, "is_open", allow_duplicate=True),
        Output(ids.TUNE_ACTIVE_DESTINATION_STORE, "data", allow_duplicate=True),
        *[
            Output(_nav.destination_view_id(name), "className", allow_duplicate=True)
            for name in _nav.DESTINATIONS
        ],
        *[
            Output(_nav.destination_button_id(name), "className", allow_duplicate=True)
            for name in _nav.DESTINATIONS
        ],
        Input(ids.TUNE_BTN_RUN_PICKER_CONFIRM, "n_clicks"),
        State(ids.TUNE_RUN_PICKER_BROWSE_DIR, "data"),
        prevent_initial_call=True,
    )
    def _confirm_run_bind(n_clicks, browsed):  # type: ignore[no-untyped-def]
        if not n_clicks:
            return (no_update,) * 12
        payload, note = discover_run_payload(sandbox, browsed or "")
        if payload is None:
            # Keep the store / body / label as-is and leave the modal open so the
            # user can pick a different directory; show the clear failure note.
            return no_update, no_update, no_update, note, no_update, *([no_update] * 7)
        # Success: re-discover (cheap — markers only, never optuna) to build the
        # loaded body, then write the store + swap the body + label, clear the
        # note, and close the modal. ``discover`` already succeeded inside
        # ``discover_run_payload``, so this re-read does not raise.
        from pathlib import Path

        root = TuneRunRoot.discover(Path(payload["path"]))
        body = build_loaded_body(root, sandbox=sandbox)
        active: _nav.Destination = "monitor"
        view_classes = [
            _nav.destination_view_class(name, active) for name in _nav.DESTINATIONS
        ]
        button_classes = [
            _nav.destination_button_class(name, active) for name in _nav.DESTINATIONS
        ]
        return (
            payload,
            body,
            payload["path"],
            "",
            False,
            active,
            *view_classes,
            *button_classes,
        )


def _register_launch_command_mirror(app) -> None:  # type: ignore[no-untyped-def]
    """Wire the clientside Launch-command mirror (Task C1).

    A single clientside callback re-renders the live command card from the
    Launch form (strategy / trials / storage-URL / screen / slurm) plus the
    hidden paths store, mirroring the pure
    :func:`~phenotypic.gui.tune._command.render_launch_command` (the unit-tested
    source-of-truth) in the browser. The mirror only builds a string — it never
    spawns a process (the no-re-optimize lock).
    """
    from dash import Input, Output, State

    app.clientside_callback(
        "function(strategy, nTrials, storageUrl, screen, slurm, paths) { "
        "return window.dash_clientside.tune_launch.renderCommand("
        "strategy, nTrials, storageUrl, screen, slurm, paths); }",
        Output(ids.TUNE_LAUNCH_COMMAND, "children"),
        Input(ids.TUNE_LAUNCH_STRATEGY, "value"),
        Input(ids.TUNE_LAUNCH_N_TRIALS, "value"),
        Input(ids.TUNE_LAUNCH_STORAGE_URL, "value"),
        Input(ids.TUNE_LAUNCH_SCREEN, "value"),
        Input(ids.TUNE_LAUNCH_SLURM, "value"),
        State(ids.TUNE_LAUNCH_PATHS_STORE, "data"),
        prevent_initial_call=True,
    )


# ---------------------------------------------------------------------------
# Space — export the edited search space to tuning_spec.json (Task C2)
# ---------------------------------------------------------------------------

def _collect_space_edits(
    keys,
    lows,
    highs,
    logs,
    choices,
    tunables,
    *,
    low_keys=None,
    high_keys=None,
    log_keys=None,
    choice_keys=None,
) -> "dict[str, dict]":
    """Zip the pattern-matching Space inputs into a per-key edit map.

    Dash hands pattern-matching ``State`` values back as positional lists aligned
    by their component ``id`` order; the matching ``ctx.states_list`` carries each
    id so we recover the knob key. Each edit dict carries the populated subset of
    ``{low, high, log, choices, tunable}`` — a knob whose widget was untouched
    yields an empty dict, which :func:`~phenotypic.gui.tune._space._apply_edits`
    treats as "use the inferred default".

    Args:
        keys: The per-knob keys (in widget order), recovered from the ids.
        lows / highs: The range low / high numeric input values.
        logs / tunables: The log / tunable switch value lists (``["on"]`` or ``[]``).
        choices: The categorical checklist value lists.

    Returns:
        ``{knob_key: {"low": …, "tunable": bool, …}}`` for every knob.
    """
    def _keyed_values(component_ids, values):  # type: ignore[no-untyped-def]
        if component_ids is None:
            return None
        return {
            component_id.get("key"): value
            for component_id, value in zip(component_ids, values)
            if isinstance(component_id, dict)
        }

    low_by_key = _keyed_values(low_keys, lows)
    high_by_key = _keyed_values(high_keys, highs)
    log_by_key = _keyed_values(log_keys, logs)
    choice_by_key = _keyed_values(choice_keys, choices)

    def _value(index, key, values, keyed):  # type: ignore[no-untyped-def]
        if keyed is not None:
            return keyed.get(key)
        return values[index] if index < len(values) else None

    edits: "dict[str, dict]" = {}
    for index, key in enumerate(keys):
        if key is None:
            continue
        edit: dict = {}
        low = _value(index, key, lows, low_by_key)
        high = _value(index, key, highs, high_by_key)
        log = _value(index, key, logs, log_by_key)
        choice = _value(index, key, choices, choice_by_key)
        tunable = tunables[index] if index < len(tunables) else None
        if low is not None:
            edit["low"] = low
        if high is not None:
            edit["high"] = high
        if log is not None:
            edit["log"] = "on" in (log or [])
        if choice is not None:
            edit["choices"] = list(choice)
        if tunable is not None:
            edit["tunable"] = "on" in (tunable or [])
        edits[key] = edit
    return edits


def write_space_spec(
    root: "TuneRunRoot", edits: "dict[str, dict]"
) -> "Path":
    """Build the edited search space and write it to ``tuning_spec.json``.

    Loads the run's existing spec (preferred) or base pipeline, rebuilds the
    :class:`~phenotypic.tune.TuningSpec` via the pure
    :func:`~phenotypic.gui.tune._space.space_to_spec` (preserving the run's
    scorer / strategy / budget when a spec already exists — OQ8), and writes the
    result **atomically** (temp file + ``os.replace``) to
    ``deliverables/tuning_spec.json``. Never spawns a run — it only persists the
    recipe (the no-re-optimize lock).

    Args:
        root: The validated tune output handle.
        edits: The per-knob edit map (empty → the inferred defaults).

    Returns:
        The path written (``deliverables/tuning_spec.json`` under the run dir).

    Raises:
        ValueError: When the run dir holds neither a spec nor a pipeline to infer
            a search space from.
        PermissionError: When the output directory is read-only (HPCC) and the
            atomic ``os.replace`` cannot complete. Re-raised so the caller can
            surface it in a note.
    """
    from phenotypic.gui.tune._space import _load_space_source, space_to_spec
    from phenotypic.sdk_ import atomic_write_text, tuning_spec_path

    source = _load_space_source(root)
    if source is None:
        raise ValueError(
            "no tuning_spec.json or pipeline.json found for this run; "
            "cannot infer a search space to export"
        )
    spec = space_to_spec(source, edits=edits)
    payload = spec.model_dump_json(indent=2)

    target = tuning_spec_path(root.path)
    # Atomic write (shared helper): temp file + ``os.replace`` so a reader never
    # sees a half-written spec. A read-only output dir (HPCC) raises
    # PermissionError, re-raised so the caller can surface it in the Space note.
    atomic_write_text(target, payload)
    return target


def _register_space_export(app) -> None:  # type: ignore[no-untyped-def]
    """Wire the Space "Export tuning_spec.json" button (Task C2).

    Collects the per-knob edits from the pattern-matching Space inputs, rebuilds
    + writes the spec via :func:`write_space_spec`, and reports the outcome in the
    Space note. The write is the only side effect — no run is spawned.
    """
    from dash import ALL, Input, Output, State

    @app.callback(
        Output(ids.TUNE_SPACE_NOTE, "children"),
        Input(ids.TUNE_BTN_SPACE_EXPORT, "n_clicks"),
        State({"type": ids.TUNE_SPACE_TUNABLE, "key": ALL}, "id"),
        State({"type": ids.TUNE_SPACE_LOW, "key": ALL}, "value"),
        State({"type": ids.TUNE_SPACE_HIGH, "key": ALL}, "value"),
        State({"type": ids.TUNE_SPACE_LOG, "key": ALL}, "value"),
        State({"type": ids.TUNE_SPACE_CHOICES, "key": ALL}, "value"),
        State({"type": ids.TUNE_SPACE_TUNABLE, "key": ALL}, "value"),
        State(ids.TUNE_RUN_ROOT_STORE, "data"),
        prevent_initial_call=True,
    )
    def _export_space(  # type: ignore[no-untyped-def]
        n_clicks, tunable_ids, lows, highs, logs, choices, tunables, run_root_data
    ):
        from pathlib import Path

        from phenotypic.gui.tune._run_root import TuneRunRoot

        if not run_root_data or not run_root_data.get("path"):
            return "No run is bound -- cannot export."
        keys = [entry.get("key") for entry in (tunable_ids or [])]
        edits = _collect_space_edits(keys, lows, highs, logs, choices, tunables)
        try:
            root = TuneRunRoot.discover(Path(run_root_data["path"]))
            written = write_space_spec(root, edits)
        except PermissionError:
            return "Export failed: the output directory is read-only."
        except Exception:  # noqa: BLE001 - surface the failure in the note
            logger.warning("Space export failed", exc_info=True)
            return "Export failed -- see the server log."
        return f"Exported {written}"


def _param_importances(store: "Optional[_ReadableStore]") -> dict[str, float]:
    """The study's param importances, or ``{}`` when unavailable.

    Only the live ``OptunaStudyStore`` exposes ``param_importances`` (fANOVA);
    the parquet journal does not, so the importance figure is empty on the
    degraded path. Any failure degrades to ``{}`` so the poll never raises.
    """
    if store is None:
        return {}
    getter = getattr(store, "param_importances", None)
    if getter is None:
        return {}
    try:
        importances = getter()
    except Exception:  # noqa: BLE001 - poll must never raise
        logger.warning("param_importances read failed", exc_info=True)
        return {}
    return dict(importances) if importances else {}


# ---------------------------------------------------------------------------
# Curate — pure pin / mode helpers (no Dash; unit-tested headless)
# ---------------------------------------------------------------------------

#: The A/B pin store slot names, in fill order.
_AB_SLOTS: tuple[str, str] = ("a", "b")


def pinned_pair(clicked_trial: int, store: "Optional[dict]") -> dict:
    """Pin ``clicked_trial`` into the A/B pair, assign-A-then-B-then-re-pin.

    The pure pin logic, unit-testable without Dash. Given the clicked trial
    number and the current ``{"a": <n|None>, "b": <n|None>}`` store:

    * an empty slot A takes the trial;
    * else an empty slot B takes it;
    * else (both full) the trial re-pins into slot A — the oldest pin cycles
      out so the user can keep comparing against a held B side.

    Clicking the trial already pinned in a slot is idempotent (it does not
    duplicate the trial into the other slot).

    Args:
        clicked_trial: The shortlist-card trial number the user clicked.
        store: The current pin store (``{"a", "b"}``), or ``None`` / partial on
            first render — missing keys are treated as empty slots.

    Returns:
        The updated pin store ``{"a": <n|None>, "b": <n|None>}``.
    """
    base = store if isinstance(store, dict) else {}
    a = base.get("a")
    b = base.get("b")
    # Idempotent: clicking an already-pinned trial leaves the pair unchanged.
    if clicked_trial in (a, b):
        return {"a": a, "b": b}
    if a is None:
        return {"a": clicked_trial, "b": b}
    if b is None:
        return {"a": a, "b": clicked_trial}
    return {"a": clicked_trial, "b": b}


#: The valid Curate overlay modes (side-by-side ↔ difference).
_CURATE_MODES: frozenset[str] = frozenset({"side", "difference"})

#: The default Curate mode when the trigger is unknown / absent.
_DEFAULT_CURATE_MODE: str = "side"


def curate_mode(trigger: "Optional[str]") -> str:
    """Resolve the Curate overlay mode from the toggle's value.

    Pure, unit-testable: a known mode (``"side"`` / ``"difference"``) passes
    through; any unknown / absent value falls back to ``"side"`` so a stray
    trigger lands on the side-by-side default.

    Args:
        trigger: The mode toggle's value.

    Returns:
        One of ``"side"`` / ``"difference"``.
    """
    if trigger in _CURATE_MODES:
        return trigger  # type: ignore[return-value]
    return _DEFAULT_CURATE_MODE


# ---------------------------------------------------------------------------
# Curate — sandbox-bounded Image Source picker (B-IMG)
# ---------------------------------------------------------------------------

#: The pre-selection prompt shown above the (empty) overlay area until an
#: Image Source is bound.
_IMAGE_SOURCE_PLACEHOLDER: str = "no Image Source selected"


def _register_curate_callbacks(app, sandbox) -> None:  # type: ignore[no-untyped-def]
    """Register the Curate-view callbacks (Image Source picker; B-IMG).

    Args:
        app: The :class:`dash.Dash` instance.
        sandbox: The frozen-at-launch sandbox bounding plate loads.
    """
    from dash import ALL, Input, Output, State, no_update

    from phenotypic.gui.tune._image_source import (
        render_image_source_tree,
        resolve_image_source,
    )

    # --- Open / cancel the picker modal -----------------------------------
    @app.callback(
        Output(ids.TUNE_IMAGE_SOURCE_MODAL, "is_open", allow_duplicate=True),
        Input(ids.TUNE_BTN_PICK_IMAGE_SOURCE, "n_clicks"),
        Input(ids.TUNE_BTN_IMAGE_SOURCE_CANCEL, "n_clicks"),
        prevent_initial_call=True,
    )
    def _toggle_image_source_modal(open_clicks, cancel_clicks):  # type: ignore[no-untyped-def]
        # The open button opens; cancel closes. Dispatch on the trigger id.
        if ctx.triggered_id == ids.TUNE_BTN_PICK_IMAGE_SOURCE and open_clicks:
            return True
        if ctx.triggered_id == ids.TUNE_BTN_IMAGE_SOURCE_CANCEL and cancel_clicks:
            return False
        return no_update

    # --- Navigate the tree (folder click → browse-dir store) --------------
    @app.callback(
        Output(ids.TUNE_IMAGE_SOURCE_BROWSE_DIR, "data", allow_duplicate=True),
        Input(
            {"type": ids.TUNE_DIR_ENTRY_IMAGE_SOURCE, "kind": ALL, "path": ALL},
            "n_clicks",
        ),
        prevent_initial_call=True,
    )
    def _navigate_image_source_tree(_clicks):  # type: ignore[no-untyped-def]
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return no_update
        if triggered.get("type") != ids.TUNE_DIR_ENTRY_IMAGE_SOURCE:
            return no_update
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return no_update
        path = triggered.get("path")
        return path if isinstance(path, str) else no_update

    # --- Re-render the tree body on browse-dir change ---------------------
    @app.callback(
        Output(ids.TUNE_IMAGE_SOURCE_MODAL_BODY, "children"),
        Input(ids.TUNE_IMAGE_SOURCE_BROWSE_DIR, "data"),
        prevent_initial_call=True,
    )
    def _render_image_source_body(dir_value):  # type: ignore[no-untyped-def]
        from pathlib import Path

        current = Path(dir_value) if dir_value else None
        return render_image_source_tree(sandbox, current)

    # --- Confirm → resolve + commit the Image Source ----------------------
    @app.callback(
        Output(ids.TUNE_IMAGE_SOURCE_STORE, "data", allow_duplicate=True),
        Output(ids.TUNE_IMAGE_SOURCE_LABEL, "children", allow_duplicate=True),
        Output(ids.TUNE_IMAGE_SOURCE_MODAL, "is_open", allow_duplicate=True),
        Output(ids.TUNE_CURATE_TOAST, "is_open", allow_duplicate=True),
        Output(ids.TUNE_CURATE_TOAST, "children", allow_duplicate=True),
        Input(ids.TUNE_BTN_IMAGE_SOURCE_CONFIRM, "n_clicks"),
        State(ids.TUNE_IMAGE_SOURCE_BROWSE_DIR, "data"),
        prevent_initial_call=True,
    )
    def _confirm_image_source(n_clicks, browsed):  # type: ignore[no-untyped-def]
        if not n_clicks or not browsed:
            return no_update, no_update, no_update, no_update, no_update
        resolved = resolve_image_source(sandbox, browsed)
        if resolved is None:
            return (
                no_update,
                no_update,
                no_update,
                True,
                f"Refused: {browsed} escapes the sandbox or is not a directory.",
            )
        return str(resolved), str(resolved), False, no_update, no_update

    @app.callback(
        Output(SHELL_SOURCE_IMAGE_ROOT_STORE, "data", allow_duplicate=True),
        Input(ids.TUNE_IMAGE_SOURCE_STORE, "data"),
        State(SHELL_SOURCE_IMAGE_ROOT_STORE, "data", allow_optional=True),
        prevent_initial_call=True,
    )
    def _mirror_tune_image_source_to_shared(
        image_source, current_payload
    ):  # type: ignore[no-untyped-def]
        payload = _source_payload_for_tune_image_source(
            sandbox, image_source, current_payload
        )
        return payload if payload is not None else no_update

    @app.callback(
        Output(ids.TUNE_IMAGE_SOURCE_STORE, "data", allow_duplicate=True),
        Output(ids.TUNE_IMAGE_SOURCE_LABEL, "children", allow_duplicate=True),
        Input(SHELL_SOURCE_IMAGE_ROOT_STORE, "data", allow_optional=True),
        State(ids.TUNE_IMAGE_SOURCE_STORE, "data"),
        prevent_initial_call="initial_duplicate",
    )
    def _initialize_tune_image_source_from_shared(
        shared_payload, current_image_source
    ):  # type: ignore[no-untyped-def]
        image_source = _tune_image_source_from_shared(
            sandbox, shared_payload, current_image_source
        )
        if image_source is None:
            return no_update, no_update
        return image_source, image_source

    # --- Mirror the Image Source store → prompt visibility ----------------
    @app.callback(
        Output(ids.TUNE_CURATE_PROMPT, "style"),
        Input(ids.TUNE_IMAGE_SOURCE_STORE, "data"),
        prevent_initial_call=True,
    )
    def _toggle_curate_prompt(image_source):  # type: ignore[no-untyped-def]
        return {"display": "none"} if image_source else {}

    _register_curate_overlay_callbacks(app, sandbox)


# ---------------------------------------------------------------------------
# Curate — shortlist pin + non-blocking overlay render/poll (B4)
# ---------------------------------------------------------------------------


def _list_plate_names(
    image_source: "Optional[str]", *, sandbox=None
) -> list[str]:
    """List image file names directly under ``image_source`` (sorted).

    Returns ``[]`` for an unset / unreadable source so the picker degrades to
    empty rather than raising. When ``sandbox`` is provided, the directory is
    re-confined before listing. Only depth-1 files with a known image extension
    are surfaced.
    """
    if not image_source:
        return []
    from pathlib import Path

    if sandbox is not None:
        from phenotypic.gui.tune._image_source import resolve_image_source

        resolved = resolve_image_source(sandbox, image_source)
        if resolved is None:
            return []
        directory = resolved
    else:
        directory = Path(image_source)
    try:
        names = [
            entry.name
            for entry in directory.iterdir()
            if entry.is_file() and entry.suffix.lower() in IMAGE_EXTS
        ]
    except OSError:
        return []
    return sorted(names)


def _shortlist_card_class(trial: "Optional[int]", pinned: dict) -> str:
    """The class string for a shortlist card (highlight its A / B pin slot)."""
    classes = ["tune-shortlist-card"]
    if trial is not None and trial == pinned.get("a"):
        classes.append("tune-shortlist-card-a")
    elif trial is not None and trial == pinned.get("b"):
        classes.append("tune-shortlist-card-b")
    return " ".join(classes)


def _register_curate_overlay_callbacks(app, sandbox=None) -> None:  # type: ignore[no-untyped-def]
    """Register the pin + render/poll + mode + winner Curate callbacks.

    Split out from the Image Source picker so the picker can register even when
    no shortlist exists yet (a brand-new live run). These callbacks render
    overlays **on demand** and stay non-blocking: the render callback submits to
    the :class:`OverlayCache` singleton and returns a spinner immediately; the
    ``dcc.Interval`` poll swaps in the real figure once the future resolves.

    Args:
        app: The :class:`dash.Dash` instance.
        sandbox: The frozen-at-launch sandbox, threaded into the overlay loader
            so the final plate-load path is re-confined to the sandbox.
    """
    import uuid

    from dash import ALL, Input, Output, State, no_update

    from phenotypic.gui.tune import _curate_overlays as ov

    # --- Session id: a fresh uuid on first paint --------------------------
    @app.callback(
        Output(ids.TUNE_SESSION_ID, "data"),
        Input(ids.TUNE_SESSION_ID, "data"),
        prevent_initial_call=False,
    )
    def _init_session_id(current):  # type: ignore[no-untyped-def]
        return current if current else uuid.uuid4().hex

    # --- Populate the plate picker from the Image Source directory --------
    @app.callback(
        Output(ids.TUNE_PLATE_PICKER, "options"),
        Output(ids.TUNE_PLATE_PICKER, "value"),
        Input(ids.TUNE_IMAGE_SOURCE_STORE, "data"),
        State(ids.TUNE_PLATE_PICKER, "value"),
        prevent_initial_call=False,
    )
    def _populate_plates(image_source, current):  # type: ignore[no-untyped-def]
        names = _list_plate_names(image_source, sandbox=sandbox)
        options = [{"label": n, "value": n} for n in names]
        value = current if current in names else (names[0] if names else None)
        return options, value

    # --- Pin a shortlist card into the A/B pair ---------------------------
    @app.callback(
        Output(ids.TUNE_AB_STORE, "data"),
        Output({"type": ids.TUNE_SHORTLIST_CARD, "trial": ALL}, "className"),
        Input({"type": ids.TUNE_SHORTLIST_CARD, "trial": ALL}, "n_clicks"),
        State(ids.TUNE_AB_STORE, "data"),
        State({"type": ids.TUNE_SHORTLIST_CARD, "trial": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _pin_card(_clicks, store, ids_list):  # type: ignore[no-untyped-def]
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or "trial" not in triggered:
            return no_update, no_update
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return no_update, no_update
        pinned = pinned_pair(int(triggered["trial"]), store)
        classes = [
            _shortlist_card_class(card_id.get("trial"), pinned)
            for card_id in ids_list
        ]
        return pinned, classes

    # --- Mode toggle → mode store + container visibility ------------------
    @app.callback(
        Output(ids.TUNE_CURATE_MODE_STORE, "data"),
        Output(ids.TUNE_SIDE_BY_SIDE, "className"),
        Output(ids.TUNE_DIFFERENCE, "className"),
        Input(ids.TUNE_CURATE_MODE_TOGGLE, "value"),
        prevent_initial_call=True,
    )
    def _switch_curate_mode(value):  # type: ignore[no-untyped-def]
        mode = curate_mode(value)
        side_cls = "tune-curate-sidebyside"
        diff_cls = "tune-curate-difference"
        if mode == "difference":
            side_cls += " tune-view-hidden"
        else:
            diff_cls += " tune-view-hidden"
        return mode, side_cls, diff_cls

    # --- Render (NON-BLOCKING): submit overlays, return spinners ----------
    @app.callback(
        Output(ids.TUNE_GRAPH_A, "figure", allow_duplicate=True),
        Output(ids.TUNE_GRAPH_B, "figure", allow_duplicate=True),
        Output(ids.TUNE_GRAPH_DIFF, "figure", allow_duplicate=True),
        Input(ids.TUNE_AB_STORE, "data"),
        Input(ids.TUNE_PLATE_PICKER, "value"),
        Input(ids.TUNE_CURATE_MODE_STORE, "data"),
        State(ids.TUNE_SESSION_ID, "data"),
        State(ids.TUNE_IMAGE_SOURCE_STORE, "data"),
        State(ids.TUNE_RUN_ROOT_STORE, "data"),
        prevent_initial_call=True,
    )
    def _render_overlays(  # type: ignore[no-untyped-def]
        pinned, plate, mode, session_id, image_source, run_root_data
    ):
        return _submit_curate_overlays(
            ov,
            pinned=pinned,
            plate=plate,
            mode=mode,
            session_id=session_id,
            image_source=image_source,
            run_root_data=run_root_data,
            sandbox=sandbox,
        )

    # --- Overlay-readiness poll: swap spinners for resolved figures -------
    @app.callback(
        Output(ids.TUNE_GRAPH_A, "figure", allow_duplicate=True),
        Output(ids.TUNE_GRAPH_B, "figure", allow_duplicate=True),
        Output(ids.TUNE_GRAPH_DIFF, "figure", allow_duplicate=True),
        Input(ids.TUNE_OVERLAY_POLL, "n_intervals"),
        State(ids.TUNE_AB_STORE, "data"),
        State(ids.TUNE_PLATE_PICKER, "value"),
        State(ids.TUNE_CURATE_MODE_STORE, "data"),
        State(ids.TUNE_SESSION_ID, "data"),
        State(ids.TUNE_RUN_ROOT_STORE, "data"),
        prevent_initial_call=True,
    )
    def _poll_overlays(  # type: ignore[no-untyped-def]
        _n, pinned, plate, mode, session_id, run_root_data
    ):
        return _poll_curate_overlays(
            ov,
            pinned=pinned,
            plate=plate,
            mode=mode,
            session_id=session_id,
            run_root_data=run_root_data,
        )

    # --- Set as winner: write deliverables/best_pipeline.json (atomic) ----
    @app.callback(
        Output(ids.TUNE_WINNER_NOTE, "children"),
        Output(ids.TUNE_CURATE_TOAST, "is_open", allow_duplicate=True),
        Output(ids.TUNE_CURATE_TOAST, "children", allow_duplicate=True),
        Input(ids.TUNE_BTN_SET_WINNER, "n_clicks"),
        State(ids.TUNE_AB_STORE, "data"),
        State(ids.TUNE_RUN_ROOT_STORE, "data"),
        prevent_initial_call=True,
    )
    def _set_winner(n_clicks, pinned, run_root_data):  # type: ignore[no-untyped-def]
        return _write_curate_winner(ov, n_clicks, pinned, run_root_data)

    _register_linked_zoom(app)


def _write_curate_winner(ov, n_clicks, pinned, run_root_data):  # type: ignore[no-untyped-def]
    """Write the A-pinned candidate as the winner; surface errors in a toast.

    The winner is slot A (the primary pin). Re-discovers the run + base
    pipeline, builds + atomically writes ``deliverables/best_pipeline.json`` via
    :func:`~phenotypic.gui.tune._winner.write_winner`, and catches
    ``PermissionError`` (OQ7 — HPCC read-only dirs) → a danger toast.
    """
    from pathlib import Path

    from dash import no_update

    from phenotypic.gui.tune._run_root import TuneRunRoot
    from phenotypic.gui.tune._winner import write_winner

    if not n_clicks:
        return no_update, no_update, no_update
    pinned = pinned if isinstance(pinned, dict) else {}
    a_trial = pinned.get("a")
    if a_trial is None:
        return no_update, True, "Pin a candidate to slot A first."
    if not (run_root_data and run_root_data.get("path")):
        return no_update, True, "No bound run."

    try:
        root = TuneRunRoot.discover(Path(run_root_data["path"]))
    except Exception:  # noqa: BLE001 - surface a friendly message, never raise
        logger.warning("Set-winner re-discovery failed", exc_info=True)
        return no_update, True, "Could not re-read the run."

    base = ov.read_base_pipeline(root)
    if base is None:
        return no_update, True, "Base pipeline unavailable -- cannot build winner."
    trials = _trials_by_number(root)
    winner = trials.get(a_trial)
    if winner is None:
        return no_update, True, f"Trial {a_trial} not in the journal."

    try:
        written = write_winner(root, base, winner)
    except PermissionError:
        logger.warning("Winner write refused (read-only output dir)", exc_info=True)
        return (
            no_update,
            True,
            "Could not write best_pipeline.json -- output directory is read-only.",
        )
    except Exception:  # noqa: BLE001 - any write error surfaces, never raises
        logger.warning("Winner write failed", exc_info=True)
        return no_update, True, "Could not write best_pipeline.json (see logs)."

    return f"Winner: trial {a_trial} → {written.name}", no_update, no_update


def _register_linked_zoom(app) -> None:  # type: ignore[no-untyped-def]
    """Wire the clientside A<->B linked-zoom mirror (Task B4).

    Two clientside callbacks (A->B and B->A), each passing its OWN graph's
    relayout prop-id as the third arg so the JS ``mirrorRange`` propagates only
    from the user-driven graph (the triggered-prop guard that stops the
    A->B->A infinite relayout). The partner graph's current ``figure`` is a
    ``State`` so the JS can clone it with the synced axis range.
    """
    from dash import Input, Output, State

    app.clientside_callback(
        f"function(r, fig) {{ return window.dash_clientside.tune_sync.mirrorRange("
        f"r, fig, '{ids.TUNE_GRAPH_A}.relayoutData'); }}",
        Output(ids.TUNE_GRAPH_B, "figure", allow_duplicate=True),
        Input(ids.TUNE_GRAPH_A, "relayoutData"),
        State(ids.TUNE_GRAPH_B, "figure"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        f"function(r, fig) {{ return window.dash_clientside.tune_sync.mirrorRange("
        f"r, fig, '{ids.TUNE_GRAPH_B}.relayoutData'); }}",
        Output(ids.TUNE_GRAPH_A, "figure", allow_duplicate=True),
        Input(ids.TUNE_GRAPH_B, "relayoutData"),
        State(ids.TUNE_GRAPH_A, "figure"),
        prevent_initial_call=True,
    )


def _submit_curate_overlays(  # type: ignore[no-untyped-def]
    ov,
    *,
    pinned,
    plate,
    mode,
    session_id,
    image_source,
    run_root_data,
    sandbox=None,
):
    """Submit the needed overlays (non-blocking) and return spinner figures.

    Pure orchestration over the overlay module ``ov`` so it is testable without
    Dash: resolves the run + base pipeline, and for each needed slot submits a
    render future (or returns a guidance figure when prerequisites are missing).
    ``sandbox`` is threaded into the loader so the plate-load path is re-confined.
    """
    from pathlib import Path

    from phenotypic.gui.tune import _curate as curate
    from phenotypic.gui.tune._overlays import get_overlay_cache
    from phenotypic.gui.tune._run_root import TuneRunRoot

    spinner = curate.placeholder_figure("rendering…")
    no_source = curate.placeholder_figure("pick an Image Source")
    no_plate = curate.placeholder_figure("pick a plate")

    if not image_source:
        return no_source, no_source, no_source
    if not plate:
        return no_plate, no_plate, no_plate
    if not (run_root_data and run_root_data.get("path")):
        return _no_update_triple()

    pinned = pinned if isinstance(pinned, dict) else {}
    a_trial, b_trial = pinned.get("a"), pinned.get("b")
    mode = curate_mode(mode)
    session = session_id or "default"

    try:
        root = TuneRunRoot.discover(Path(run_root_data["path"]))
    except Exception:  # noqa: BLE001 - render must degrade, never raise
        logger.warning("Curate render re-discovery failed", exc_info=True)
        return _no_update_triple()

    base = ov.read_base_pipeline(root)
    if base is None:
        unavailable = curate.placeholder_figure("base pipeline unavailable")
        return unavailable, unavailable, unavailable

    cache = get_overlay_cache(root.path)
    trials = _trials_by_number(root)

    fig_a = _submit_one_candidate(
        ov, cache, base, trials, session, a_trial, plate, image_source, spinner,
        sandbox,
    )
    fig_b = _submit_one_candidate(
        ov, cache, base, trials, session, b_trial, plate, image_source, spinner,
        sandbox,
    )
    fig_diff = _submit_difference(
        ov, cache, base, trials, session, a_trial, b_trial, plate, image_source,
        spinner, sandbox,
    )
    return fig_a, fig_b, fig_diff


def _no_update_triple():  # type: ignore[no-untyped-def]
    """Three ``no_update`` sentinels for the three Curate graph outputs."""
    from dash import no_update

    return no_update, no_update, no_update


def _trials_by_number(root: "TuneRunRoot") -> "dict[int, object]":
    """Map ``trial.number -> Trial`` from the run's journal (``{}`` when none)."""
    store = _load_journal(root)
    if store is None:
        return {}
    return {t.number: t for t in store.trials}


def _submit_one_candidate(  # type: ignore[no-untyped-def]
    ov, cache, base, trials, session, trial_number, plate, image_source, spinner,
    sandbox=None,
):
    """Submit one candidate overlay; return its spinner / guidance figure."""
    from phenotypic.gui.tune import _curate as curate

    if trial_number is None:
        return curate.placeholder_figure("pin a candidate")
    trial = trials.get(trial_number)
    if trial is None:
        return curate.placeholder_figure(f"trial {trial_number} not in journal")
    key = ov.candidate_key(session, trial_number, plate)

    def _render():  # type: ignore[no-untyped-def]
        from phenotypic.gui.tune._overlays import render_candidate_overlay

        grid = ov.load_plate_grid(image_source, plate, sandbox=sandbox)
        return render_candidate_overlay(base, trial.params, grid)

    ov.request_overlay(cache, key, _render)
    return spinner


def _submit_difference(  # type: ignore[no-untyped-def]
    ov, cache, base, trials, session, a_trial, b_trial, plate, image_source, spinner,
    sandbox=None,
):
    """Submit the A-vs-B difference overlay; return its spinner / guidance."""
    from phenotypic.gui.tune import _curate as curate

    if a_trial is None or b_trial is None:
        return curate.placeholder_figure("pin A and B to diff")
    trial_a = trials.get(a_trial)
    trial_b = trials.get(b_trial)
    if trial_a is None or trial_b is None:
        return curate.placeholder_figure("a pinned trial is not in the journal")
    key = ov.difference_key(session, a_trial, b_trial, plate)

    def _render():  # type: ignore[no-untyped-def]
        from phenotypic.gui.tune._overlays import OVERLAY_MAX_DIM, render_difference
        from phenotypic.tune._evaluation._builder import build_pipeline

        grid = ov.load_plate_grid(image_source, plate, sandbox=sandbox)
        seg_a = build_pipeline(base, trial_a.params).apply(grid.copy())
        seg_b = build_pipeline(base, trial_b.params).apply(grid.copy())
        # Clamp to the same max_dim as the candidate overlay so the full-res
        # plate is never serialized to the browser or cached at full res.
        return render_difference(
            grid.rgb[:], seg_a.objmap[:], seg_b.objmap[:], max_dim=OVERLAY_MAX_DIM
        )

    ov.request_overlay(cache, key, _render)
    return spinner


def _poll_curate_overlays(  # type: ignore[no-untyped-def]
    ov, *, pinned, plate, mode, session_id, run_root_data=None
):
    """Swap any resolved overlay into its figure; else ``no_update``.

    Two-tier resolution per slot (the B4 self-heal): first the in-flight
    ``_PENDING`` future (``overlay_ready`` → ``take_overlay``); if no future is
    ready/available for a slot, fall back to a **non-consuming**
    :meth:`OverlayCache.peek` of the same cache key. The OverlayCache is
    authoritative — the rendered array lives there independent of the per-tab
    future registry — so a future dropped by a re-submit (a sibling pin's
    stale-drop) or already consumed by an earlier poll tick self-heals into the
    cached figure instead of wedging on a permanent "rendering…" spinner.

    ``run_root_data`` resolves the per-run cache for the peek. When it is missing
    / un-discoverable the peek tier is simply skipped (degrades to take-only, the
    pre-fix behaviour) — the poll never raises.
    """
    from phenotypic.gui.tune import _curate as curate

    pinned = pinned if isinstance(pinned, dict) else {}
    a_trial, b_trial = pinned.get("a"), pinned.get("b")
    session = session_id or "default"
    error_fig = curate.placeholder_figure("render failed -- see logs")

    cache = _resolve_overlay_cache(run_root_data)

    def _swap(key):  # type: ignore[no-untyped-def]
        from dash import no_update

        if key is None:
            return no_update
        # Tier 1: an in-flight future that has resolved → consume it.
        if ov.overlay_ready(key):
            array = ov.take_overlay(key)
            if array is not None:
                return ov.overlay_figure(array)
            # A resolved-to-None future is a genuine render failure (the cache
            # stores nothing for it), so surface the error rather than spin.
            return error_fig
        # Tier 2 (self-heal): no live future, but the array may already be in the
        # authoritative cache (future dropped on a re-submit or consumed earlier).
        if cache is not None:
            cached = cache.peek(ov.cache_key_for(key))
            if cached is not None:
                return ov.overlay_figure(cached)
        return no_update

    key_a = ov.candidate_key(session, a_trial, plate) if a_trial is not None and plate else None
    key_b = ov.candidate_key(session, b_trial, plate) if b_trial is not None and plate else None
    key_diff = (
        ov.difference_key(session, a_trial, b_trial, plate)
        if a_trial is not None and b_trial is not None and plate
        else None
    )
    return _swap(key_a), _swap(key_b), _swap(key_diff)


def _resolve_overlay_cache(run_root_data):  # type: ignore[no-untyped-def]
    """Resolve the per-run :class:`OverlayCache` for the poll's self-heal peek.

    Returns ``None`` (never raises) when no run is bound or the run path can't be
    re-discovered, so the poll degrades to take-only resolution.
    """
    if not (run_root_data and run_root_data.get("path")):
        return None
    from pathlib import Path

    from phenotypic.gui.tune._overlays import get_overlay_cache
    from phenotypic.gui.tune._run_root import TuneRunRoot

    try:
        root = TuneRunRoot.discover(Path(run_root_data["path"]))
        return get_overlay_cache(root.path)
    except Exception:  # noqa: BLE001 - poll must never raise; peek is optional
        logger.warning("Overlay poll cache resolution failed", exc_info=True)
        return None


__all__ = ["active_view", "read_study_for_monitor", "register_callbacks"]
