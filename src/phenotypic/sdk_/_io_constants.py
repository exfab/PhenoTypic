"""Shared CLI ↔ GUI artifact-layout constants.

Single source of truth for everything written to disk by the forward CLI,
the SLURM recompile worker, and the chunk writer — and read back by the
GUI viewer / analysis sub-app. Both subpackages should import from here
rather than re-spelling. This module replaces the scattered inline
``"master_measurements.parquet"`` / ``"progress"`` / etc. literals that
previously lived as private constants in ~10 different files.

Module layout
-------------
* **Filenames** (`MASTER_MEASUREMENTS_*`, `PIPELINE_JSON`, …) — bare strings
  written/read by both producer (CLI) and consumer (GUI).
* **Directory names** (`DIR_*`) — same.
* **Templated filenames** — private ``_FOO_TEMPLATE: Final[str]``
  constants paired with public ``foo_filename(...)`` render functions.
  The render function's typed signature is the public API; the template
  itself is private. This is the project's first deployment of the
  CLAUDE.md "parameterized strings are not enumerations" pattern.
* **Path helpers** — `progress_dir(output)`, `event_log_path(output)`,
  etc. Take a base ``output_dir: Path`` and return the canonical
  artifact path. Replace ``output_dir / "literal"`` constructions
  scattered across the CLI. Two parameter conventions are used:

    - **`output_dir: Path`** — the run output root (e.g. ``./out``).
      Use these from any caller that has the run root in scope:
      ``master_measurements_csv_path``, ``manifest_json_path``,
      ``job_metadata_path``, ``pipeline_json_path``, ``task_status_path``,
      ``logs_dir``, ``slurm_scripts_dir``, ``analysis_html_path``,
      ``processing_report_html_path``, ``measurements_by_feature_dir``,
      etc.
    - **`progress_dir_: Path`** — the already-resolved progress dir
      (i.e. ``output_dir / "progress"``). Used for helpers that produce
      paths to *internal mid-run artifacts* the SLURM sentinel + chunk
      writer hand around between each other:
      ``analysis_full_parquet_path``, ``analysis_scatter_json_path``,
      ``sentinel_resubmitted_path``, ``chunk_lock_path``, ``chunks_dir``,
      ``chunk_parquet_path``, ``checkpoint_lock_path``, ``recompile_dir``,
      ``recompile_status_dir``. The trailing underscore on the parameter
      name disambiguates it from the ``progress_dir(output_dir)``
      function.

  When in doubt, prefer the ``output_dir``-rooted form — it composes
  cleanly with ``progress_dir(output_dir)`` if you need the progress
  directory separately. Helpers that take ``progress_dir_`` are noted
  in their individual docstrings.
* **Reader helpers** — `read_run_manifest`, `load_master_measurements`,
  `resolve_execution_mode` consolidate three high-frequency duplicates.
* **JSON contract keys** (`JobMetadataKey`, `DashboardManifestKey`,
  `ChunkStateKey`, `ChunkManifestKey`, `HdfAttr`) — namespace classes
  whose class-level ``Final[str]`` attributes are the keys writers and
  readers must reference instead of bare strings. Keeps the contract
  on-disk format mechanically discoverable.
* **`ModulePath`** — importable module paths used in ``importlib.import_module``
  dispatch by the analysis GUI's recipe loader.
* **`EnvVar`** — environment variable names read by the CLI (SLURM /
  scratch).

See also
--------
:mod:`phenotypic.sdk_.constants_`
    Image-data and framework-config enums (``IMAGE_MODE``, ``IMAGE_TYPES``,
    ``GAMMA_ENCODINGS``, ``PIPE_STATUS``). The ``METADATA`` enum and the
    experimental-tag vocabulary now live in :mod:`phenotypic.schema`.
:mod:`phenotypic.sdk_.typing_`
    Literal aliases for closed value sets used at public boundaries
    (``ExecutionMode``, ``ImageTypeName``, …).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Iterable, Optional

from .typing_ import (
    CheckpointType,
    ExecutionMode,
    ImageTypeName,
)

if TYPE_CHECKING:
    import polars as pl  # type: ignore[import-not-found]

    from phenotypic._core._grid_image import GridImage as _GridImage
    from phenotypic._core._image import Image as _Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI artifact filenames
# ---------------------------------------------------------------------------

LEGACY_JSON_SUFFIX: Final[str] = ".json"
CONFIG_SUFFIX_PIPELINE: Final[str] = ".json.pht-pipe"
CONFIG_SUFFIX_OPERATION: Final[str] = ".json.pht-op"
CONFIG_SUFFIX_COLOR_CHECKER: Final[str] = ".json.pht-cc"
CONFIG_SUFFIX_TUNING: Final[str] = ".json.pht-tune"
CONFIG_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        CONFIG_SUFFIX_PIPELINE,
        CONFIG_SUFFIX_OPERATION,
        CONFIG_SUFFIX_COLOR_CHECKER,
        CONFIG_SUFFIX_TUNING,
    }
)
PIPELINE_CONFIG_SUFFIXES: Final[frozenset[str]] = frozenset(
    {CONFIG_SUFFIX_PIPELINE, LEGACY_JSON_SUFFIX}
)
TUNING_CONFIG_SUFFIXES: Final[frozenset[str]] = frozenset(
    {CONFIG_SUFFIX_TUNING, LEGACY_JSON_SUFFIX}
)


def has_config_suffix(path: str | Path, suffixes: Iterable[str]) -> bool:
    """Return whether ``path`` ends with any configured suffix.

    Matching is case-insensitive so callers can discover user-provided files
    from case-preserving filesystems without rewriting their names.
    """
    text = str(path).lower()
    return any(text.endswith(suffix.lower()) for suffix in suffixes)


def matches_any_suffix(path: str | Path, suffixes: Iterable[str]) -> bool:
    """Return whether ``path`` ends with any suffix in ``suffixes``."""
    return has_config_suffix(path, suffixes)


def ensure_typed_json_suffix(path: str | Path, suffix: str) -> Path:
    """Return ``path`` with the canonical typed JSON suffix appended.

    Bare stems receive the full typed suffix. Legacy ``.json`` paths receive
    only the typed tail, preserving the user-provided stem and case.
    """
    target = Path(path)
    text = str(target)
    lowered = text.lower()
    canonical_suffix = suffix.lower()
    if lowered.endswith(canonical_suffix):
        return target
    if lowered.endswith(LEGACY_JSON_SUFFIX):
        typed_tail = suffix.removeprefix(LEGACY_JSON_SUFFIX)
        return Path(f"{text}{typed_tail}")
    return Path(f"{text}{suffix}")


#: Master archive of all aggregated measurements (clean, pre-post). Written by
#: :func:`phenotypic._cli._cli_output_manager.aggregate_measurements` after
#: every per-image Parquet has been concatenated and joined with optional
#: external metadata. Treated as the authoritative source by downstream
#: tooling; never edited in place.
MASTER_MEASUREMENTS_CSV: Final[str] = "master_measurements.csv"

#: Parquet companion of :data:`MASTER_MEASUREMENTS_CSV`. Preserves dtypes the
#: CSV cannot and is the format the GUI viewer prefers when present.
MASTER_MEASUREMENTS_PARQUET: Final[str] = "master_measurements.parquet"

#: Editable curated CSV mirror seeded by the CLI as a copy of
#: :data:`MASTER_MEASUREMENTS_CSV` after :func:`_apply_post_to_master`. The
#: results viewer rewrites this file in place when the user removes/restores
#: colonies. Re-running the CLI overwrites it with a fresh full copy.
MEASUREMENTS_CSV: Final[str] = "measurements.csv"

#: Editable curated Parquet companion to :data:`MEASUREMENTS_CSV`. The
#: parquet is the GUI's source of truth at boot (CSV is the human-readable
#: mirror); both are written atomically together.
MEASUREMENTS_PARQUET: Final[str] = "measurements.parquet"

#: Output filename of the model-fit summary written by the CLI when the
#: pipeline has a ``model`` configured (and re-emitted by the analysis
#: GUI's "Run analysis" button). Human-readable mirror of
#: :data:`ANALYSIS_PARQUET`.
ANALYSIS_CSV: Final[str] = "analysis.csv"

#: Parquet companion of :data:`ANALYSIS_CSV` — primary downstream
#: artifact since it preserves dtypes the CSV cannot.
ANALYSIS_PARQUET: Final[str] = "analysis.parquet"

#: Canonical pipeline-spec filename written into the output root by the
#: CLI (and rewritten by the analysis GUI on every recipe edit). Captures
#: operations, measurements, post, filters, and model — i.e. the whole
#: reproducibility surface.
PIPELINE_JSON: Final[str] = f"pipeline{CONFIG_SUFFIX_PIPELINE}"
_LEGACY_PIPELINE_JSON: Final[str] = f"pipeline{LEGACY_JSON_SUFFIX}"

#: Resume-state JSON written by ``ProcessingState.save`` and read by
#: ``ProcessingState.load`` when ``--resume`` is passed.
PROCESSING_STATE_JSON: Final[str] = "processing_state.json"

#: Human-readable run README generated by the CLI's ``READMEGenerator``
#: (output-layout + measurement-schema documentation). A user-facing
#: deliverable, so it lives in :data:`DIR_DELIVERABLES` — see
#: :func:`readme_md_path`.
README_MD: Final[str] = "README.md"

#: The resolved tuning spec echoed by ``python -m phenotypic.tune``
#: into :data:`DIR_DELIVERABLES` — the self-contained, re-runnable recipe.
TUNING_SPEC_JSON: Final[str] = f"tuning_spec{CONFIG_SUFFIX_TUNING}"
_LEGACY_TUNING_SPEC_JSON: Final[str] = f"tuning_spec{LEGACY_JSON_SUFFIX}"

#: The winning best pipeline written by the tune CLI into
#: :data:`DIR_DELIVERABLES`; reloads as a runnable ``ImagePipeline``.
BEST_PIPELINE_JSON: Final[str] = f"best_pipeline{CONFIG_SUFFIX_PIPELINE}"
_LEGACY_BEST_PIPELINE_JSON: Final[str] = f"best_pipeline{LEGACY_JSON_SUFFIX}"

#: Sidecar with the selected headline trial's params and scores, written by the
#: tune CLI into :data:`DIR_DELIVERABLES` for GUI Monitor display.
BEST_PARAMS_JSON: Final[str] = "best_params.json"

#: The RF-permutation ``param_importance.json`` report written by the tune
#: CLI into :data:`DIR_DELIVERABLES`.
PARAM_IMPORTANCE_JSON: Final[str] = "param_importance.json"

#: The tune trial journal ``trials.parquet`` written at the output-dir root
#: (powers CLI resume), not under :data:`DIR_DELIVERABLES`.
TRIALS_PARQUET: Final[str] = "trials.parquet"

#: The canonical Optuna study database ``study.db`` (SQLite WAL) written inside
#: the hidden tune cache (:data:`DIR_PHT_TUNE_CACHE`). Holds the Optuna-backed
#: store's persistent, resumable sampler state when the ``tune`` extra is used.
#: A legacy run wrote it at the output root; :func:`resolve_study_db_path` reads
#: either location (no migration).
STUDY_DB: Final[str] = "study.db"

#: The robust-eval held-out split assignment ``split.json`` written inside
#: :data:`DIR_SPLITS`. A machine-state sidecar — it must survive a
#: fresh-master rewrite and gate resume — so it lives in the hidden tune cache
#: (:data:`DIR_PHT_TUNE_CACHE`), not under :data:`DIR_DELIVERABLES`. See
#: :func:`tune_cache_split_assignment_path`.
SPLIT_ASSIGNMENT_JSON: Final[str] = "split.json"

#: The tune-run marker ``run.json`` written into :data:`DIR_PHT_TUNE_CACHE` at
#: run START (before any deliverable exists), so the GUI shell classifier can
#: recognise a live or finished tune output even before ``deliverables/`` lands.
#: Carries the study identity + storage URL + run policy. See
#: :func:`tune_cache_run_marker_path`.
RUN_MARKER_JSON: Final[str] = "run.json"

#: The robust-eval generalization report ``generalization.json`` written into
#: :data:`DIR_DELIVERABLES` — a user-facing deliverable (the winner's held-out
#: gap verdict). See :func:`generalization_path`.
GENERALIZATION_JSON: Final[str] = "generalization.json"

#: The Pareto-front parquet written by a **multi-objective** tune run into
#: :data:`DIR_PARETO` (under ``deliverables/``). One row per non-dominated trial
#: (the same schema as :data:`TRIALS_PARQUET`, with ``objectives_json`` populated).
PARETO_FRONT_PARQUET: Final[str] = "pareto_front.parquet"

#: Per-objective best-pipeline filename template written into :data:`DIR_PARETO`:
#: ``best_<objective>`` with the pipeline config suffix. One per
#: objective axis — the pipeline maximizing that single objective on the front.
#: Rendered by :func:`pareto_best_pipeline_path`; kept private (a parameterized
#: string is not an enumeration — see the code-style note on render functions).
_PARETO_BEST_PIPELINE_FILENAME_TEMPLATE: Final[str] = (
    f"best_{{objective}}{CONFIG_SUFFIX_PIPELINE}"
)

#: Per-objective param-importance filename template written into :data:`DIR_PARETO`:
#: ``param_importance_<objective>.json`` (e.g. ``param_importance_s0.json``). The
#: multi-objective sibling of :data:`PARAM_IMPORTANCE_JSON` — one RF-permutation
#: importance report per objective axis. Rendered by :func:`pareto_importance_path`;
#: kept private (a parameterized string is not an enumeration).
_PARETO_IMPORTANCE_FILENAME_TEMPLATE: Final[str] = "param_importance_{objective}.json"

# ---------------------------------------------------------------------------
# QC artifact filenames (live inside DIR_QC)
# ---------------------------------------------------------------------------

#: Per-group QC summary written by
#: :func:`phenotypic.sdk_._qc_recipe._runner.run_qc`. One row per
#: ``(instance_id, groupby key)``: the worst-direction metric, tri-state
#: status, flag, member/flagged counts, and a worst-first ``rank``.
QC_SUMMARY_PARQUET: Final[str] = "qc_summary.parquet"

#: Per-group member colonies written by :func:`phenotypic.sdk_._qc_recipe._runner.run_qc`.
#: One row per ``(instance_id, group member)`` carrying the curation key
#: (``Metadata_ImageFile`` + ``Object_Label``) plus the member's
#: contributing value, so the Review gallery can render each group's tiles.
QC_MEMBERS_PARQUET: Final[str] = "qc_members.parquet"

#: Snapshot of the ``qc`` config entries (``instance_id``/``class``/
#: ``enabled``/``params``) that produced the current ``qc/`` artifact.
#: Written by :func:`phenotypic.sdk_._qc_recipe._runner.run_qc`.
QC_CONFIG_JSON: Final[str] = "qc_config.json"

#: Per-module GUI review progress (``instance_id`` -> reviewed group keys +
#: last position). Written **only** by the results-viewer QC Review tab;
#: :func:`phenotypic.sdk_._qc_recipe._runner.run_qc` never touches it. The CLI finalize
#: path clears it on every rerun (a fresh run resets review progress).
QC_REVIEW_STATE_JSON: Final[str] = "review_state.json"

# ---------------------------------------------------------------------------
# Run-time progress sidecar files (live inside DIR_PROGRESS)
# ---------------------------------------------------------------------------

#: Append-only JSONL event log of per-image processing transitions.
#: Producers: ``_cli_update_state.append_event``. Consumers: dashboard
#: generator, recompile worker. Was previously re-spelled in 10 files.
PROCESSING_EVENTS_LOG: Final[str] = "processing_events.log"

#: SLURM job-metadata sidecar written once at job submission. Keys are
#: documented in :class:`JobMetadataKey`.
JOB_METADATA_JSON: Final[str] = "job_metadata.json"

#: Append-only JSONL of per-image failures. Each row carries a
#: :data:`phenotypic.sdk_.typing_.FailureSource` tag.
FAILURES_JSONL: Final[str] = "failures.jsonl"

#: SLURM checkpoint chunk manifest (mid-run partial-result feed).
CHUNK_MANIFEST_JSON: Final[str] = "chunk_manifest.json"

#: SLURM checkpoint chunk state (which per-image files have been chunked).
CHUNK_STATE_JSON: Final[str] = "chunk_state.json"

#: Manifest of overlay PNGs (one entry per per-image overlay).
OVERLAY_MANIFEST_JSON: Final[str] = "overlay_manifest.json"

#: Top-level dashboard manifest read by the dashboard HTML JS shim.
#: Keys are documented in :class:`DashboardManifestKey`.
MANIFEST_JSON: Final[str] = "manifest.json"

#: Per-dataset pre-aggregated Parquet emitted by the chunk writer for the
#: GPFS-optimized aggregation path. Underscore prefix marks the file as
#: an aggregator-internal intermediate (chunk-writer scan should skip).
DATASET_AGGREGATED_PARQUET: Final[str] = "_dataset_aggregated.parquet"

# ---------------------------------------------------------------------------
# Dashboard / log files
# ---------------------------------------------------------------------------

#: Generated HTML dashboard with charts + chunked-data viewer.
DASHBOARD_HTML: Final[str] = "dashboard.html"

#: Per-run stdout capture written by the run console's local runner.
STDOUT_LOG: Final[str] = "stdout.log"

#: Generated standalone analysis HTML emitted alongside the dashboard.
ANALYSIS_HTML: Final[str] = "analysis.html"

#: Top-level processing report HTML emitted by recompile mode.
PROCESSING_REPORT_HTML: Final[str] = "processing_report.html"

#: Mid-run combined Parquet that the chunk writer rewrites incrementally
#: as new per-image measurements arrive (lives in ``DIR_PROGRESS``).
ANALYSIS_FULL_PARQUET: Final[str] = "analysis_full.parquet"

#: Pre-computed scatter-plot data for the dashboard analysis pane.
ANALYSIS_SCATTER_JSON: Final[str] = "analysis_scatter.json"

#: Sentinel marker file written by the SLURM sentinel after a re-submission;
#: presence prevents an infinite resubmission loop on the next checkpoint.
SENTINEL_RESUBMITTED_MARKER: Final[str] = "sentinel_resubmitted"

#: Hidden chunk-aggregation lock file (lives in ``DIR_PROGRESS``).
CHUNK_LOCK: Final[str] = ".chunk_lock"

#: Recompile task manifest JSON (one per recompile invocation; lives at
#: ``<output>/progress/recompile/task_manifest.json``). Lists every per-image
#: shard the recompile worker is responsible for; consumed by
#: ``_cli_recompile_worker._main``.
RECOMPILE_TASK_MANIFEST_JSON: Final[str] = "task_manifest.json"


# ---------------------------------------------------------------------------
# CLI artifact directory names
# ---------------------------------------------------------------------------

#: ``<output>/results/`` — per-dataset subdirectories live below here.
DIR_RESULTS: Final[str] = "results"

#: ``<output>/progress/`` — sidecar JSON / JSONL state for in-flight runs.
DIR_PROGRESS: Final[str] = "progress"

#: Per-dataset measurements subdirectory: ``<output>/results/<ds>/measurements/``.
DIR_MEASUREMENTS: Final[str] = "measurements"

#: Per-feature spreadsheet split written by
#: :func:`phenotypic._cli._cli_output_manager.split_master_by_feature`.
DIR_MEASUREMENTS_BY_FEATURE: Final[str] = "measurements_by_feature"

#: SLURM stdout/stderr / sbatch-script subdirectory.
DIR_LOGS: Final[str] = "logs"

#: HDF5 image-state subdirectory: ``<output>/results/<ds>/hdf/``.
DIR_HDF: Final[str] = "hdf"

#: Overlay PNG subdirectory: ``<output>/results/<ds>/overlays/``.
DIR_OVERLAYS: Final[str] = "overlays"

#: Inspect-figure PNG subdirectory:
#: ``<output>/results/<ds>/inspect/<measurer-step>/``. Provisioned only
#: when the CLI's ``--save-inspect`` flag is set; per-measurer
#: subdirectories are created lazily by ``OutputManager.save_inspect``.
DIR_INSPECT: Final[str] = "inspect"

#: Mid-run chunk parquet subdirectory: ``<progress>/chunks/``.
DIR_CHUNKS: Final[str] = "chunks"

#: Recompile-worker shard / status subdirectory: ``<progress>/recompile/``.
DIR_RECOMPILE: Final[str] = "recompile"

#: Per-task JSON status subdirectory: ``<progress>/recompile/status/``.
DIR_RECOMPILE_STATUS: Final[str] = "status"

#: Per-shard parquet subdirectory inside the recompile dir:
#: ``<progress>/recompile/measurement_shards/``.
DIR_RECOMPILE_SHARDS: Final[str] = "measurement_shards"

#: Generated SLURM script subdirectory: ``<output>/slurm_scripts/``.
DIR_SLURM_SCRIPTS: Final[str] = "slurm_scripts"

#: QC artifact subdirectory: ``<output>/qc/``. Holds
#: :data:`QC_SUMMARY_PARQUET`, :data:`QC_MEMBERS_PARQUET`,
#: :data:`QC_CONFIG_JSON` (written by ``run_qc``), and
#: :data:`QC_REVIEW_STATE_JSON` (written by the GUI Review tab).
DIR_QC: Final[str] = "qc"

#: ``<output>/deliverables/`` — all user-facing run outputs collected in one
#: folder: master + post-applied measurements, the per-feature splits, the
#: analysis frames, the generated dashboard/analysis/report HTML, the
#: :data:`README_MD`, and the canonical :data:`PIPELINE_JSON`. Distinct from
#: machine-state sidecars (``progress/``, ``processing_state.json``) and from
#: per-image artifacts (``results/``), which stay at the output root. Every
#: artifact-path helper below that previously rooted at ``<output>/`` now
#: roots at :func:`deliverables_dir`.
DIR_DELIVERABLES: Final[str] = "deliverables"

#: Per-category error-object parquet subdirectory under deliverables:
#: ``<output>/deliverables/errors/<category>.parquet``. Holds the master rows
#: for each triaged error category. **Dual-owned:** the GUI writes them live as
#: the user curates (via ``CurationLabels._save_locked``) and CLI finalize
#: re-emits them (all categories) from the durable ``qc/curation_labels.parquet``
#: via ``reemit_error_deliverables`` — so headless == live.
DIR_ERRORS: Final[str] = "errors"

#: Durable curation-labels store: ``<output>/qc/curation_labels.parquet``.
#: The source of truth for categorized removals; the CLI re-keys but never
#: wipes it (contrast :data:`QC_REVIEW_STATE_JSON`).
CURATION_LABELS_PARQUET: Final[str] = "curation_labels.parquet"

#: Ordered custom-category registry sidecar: ``<output>/qc/custom_categories.json``.
CUSTOM_CATEGORIES_JSON: Final[str] = "custom_categories.json"

#: Ranked error-cutoff analysis deliverables (``ErrorCutoffFinder`` output;
#: columns ``[category, *RESULT_COLUMNS]``). **Dual-owned, but with differing
#: scope:** the GUI's Error tab writes the parquet/csv live for the *focused*
#: category (transient, last-viewed-in-session) on each recompute, while CLI
#: finalize (``reemit_error_deliverables``) authoritatively rewrites them across
#: *all* labeled categories from the durable labels store. The HTML is written
#: only on an explicit GUI ``Save analysis report`` or by CLI finalize — never on
#: a live recompute.
ERROR_ANALYSIS_PARQUET: Final[str] = "error_analysis.parquet"
ERROR_ANALYSIS_CSV: Final[str] = "error_analysis.csv"
ERROR_ANALYSIS_HTML: Final[str] = "error_analysis.html"

#: Filename of the GUI-written verified-good baseline archive (spec §9). It is
#: derived from ``qc/review_state.json`` (which CLI finalize RESETS), so it is
#: GUI-owned and never CLI-emitted; finalize leaves any existing file untouched.
VERIFIED_PARQUET: Final[str] = "verified.parquet"

#: ``splits/`` — the robust-eval held-out **split assignment** sidecar folder
#: (holds :data:`SPLIT_ASSIGNMENT_JSON`). Lives inside the hidden tune cache
#: (:data:`DIR_PHT_TUNE_CACHE`), **not** under :data:`DIR_DELIVERABLES`: the
#: split is machine state that must survive a fresh-master rewrite and gate
#: resume. See :func:`tune_cache_splits_dir`.
DIR_SPLITS: Final[str] = "splits"

#: ``<output>/deliverables/pareto/`` — the **multi-objective** sub-folder holding
#: the Pareto front parquet (:data:`PARETO_FRONT_PARQUET`) and the per-objective
#: best pipelines (:func:`pareto_best_pipeline_path`). Written only by a
#: multi-objective tune run; a single-objective run never creates it (the
#: back-compat lock — plan §0b). Rooted under :func:`deliverables_dir`.
DIR_PARETO: Final[str] = "pareto"

#: ``<output>/.phenotypic/`` — hidden machine-state cache root. Holds the
#: run's progress/, processing_state.json, and processing_events.log. Hidden
#: so it does not clutter the user-facing output folder and is skipped by the
#: GUI's run/dataset candidate scan. NOTE: distinct from the GUI's
#: ``.phenotypic-gui`` sandbox dir (presets/state) — different root, different
#: purpose.
DIR_PHENOTYPIC: Final[str] = ".phenotypic"

#: ``<output>/.pht-tune-cache/`` — hidden machine-state cache root for a **tune**
#: run, the sibling of :data:`DIR_PHENOTYPIC` for the forward CLI. Holds the
#: Optuna ``study.db`` (+ WAL), the held-out ``splits/split.json``, and the
#: GUI-discovery ``run.json`` marker. Hidden so it does not clutter the
#: user-facing output and is skipped by the GUI candidate scan; the tune cache
#: is kept distinct from ``.phenotypic`` so a directory that hosts both a
#: forward run and a tune run never collides their machine-state. Note that
#: ``trials.parquet`` is NOT relocated here — it stays at the output root as the
#: dual-purpose Optuna-resume + user-facing trial journal.
DIR_PHT_TUNE_CACHE: Final[str] = ".pht-tune-cache"


# ---------------------------------------------------------------------------
# Templated filenames — Final[str] template + typed render function
# ---------------------------------------------------------------------------
# Pattern: keep the raw ``"…{x}…"`` template private as ``_FOO_TEMPLATE``,
# expose a typed render function whose parameters are the public API.
# Callers must reach the template through the function — never .format() it
# directly. This is the project's first deployment of CLAUDE.md's
# "parameterized strings are not enumerations" rule.

_TASK_STATUS_FILENAME_TEMPLATE: Final[str] = "task_{task_index}.json"


def task_status_filename(task_index: int) -> str:
    """Filename of a per-task SLURM-recompile status JSON.

    Args:
        task_index: Zero-based recompile task index.

    Returns:
        Filename relative to ``<progress>/recompile/status/``.
    """
    return _TASK_STATUS_FILENAME_TEMPLATE.format(task_index=task_index)


_SHARD_PARQUET_FILENAME_TEMPLATE: Final[str] = "shard_{shard_id}.parquet"


def shard_parquet_filename(shard_id: int) -> str:
    """Filename of a per-shard Parquet inside the recompile worker.

    Args:
        shard_id: Zero-based shard index.

    Returns:
        Filename relative to the recompile shard directory.
    """
    return _SHARD_PARQUET_FILENAME_TEMPLATE.format(shard_id=shard_id)


_CHUNK_PARQUET_FILENAME_TEMPLATE: Final[str] = "chunk_{chunk_id:03d}.parquet"


def chunk_parquet_filename(chunk_id: int) -> str:
    """Filename of a dashboard chunk Parquet (zero-padded chunk id).

    Args:
        chunk_id: Zero-based chunk index, formatted ``{chunk_id:03d}``.

    Returns:
        Filename relative to ``<progress>/chunks/``.
    """
    return _CHUNK_PARQUET_FILENAME_TEMPLATE.format(chunk_id=chunk_id)


_DEFAULT_OUTPUT_DIR_NAME_TEMPLATE: Final[str] = "phenotypic_results_{timestamp}"
_DEFAULT_OUTPUT_TIMESTAMP_FORMAT: Final[str] = "%Y%m%d_%H%M%S"


def default_output_dir_name(now: Optional[datetime] = None) -> str:
    """Default name for an auto-generated output directory.

    Legacy helper for timestamped output-directory names.

    Args:
        now: Override clock for tests; defaults to ``datetime.now()``.

    Returns:
        ``"phenotypic_results_YYYYMMDD_HHMMSS"``.
    """
    when = now if now is not None else datetime.now()
    return _DEFAULT_OUTPUT_DIR_NAME_TEMPLATE.format(
        timestamp=when.strftime(_DEFAULT_OUTPUT_TIMESTAMP_FORMAT)
    )


_CHECKPOINT_LOCK_FILENAME_TEMPLATE: Final[str] = ".{checkpoint_type}_lock"


def checkpoint_lock_filename(checkpoint_type: CheckpointType) -> str:
    """Filename of the SLURM-sentinel exclusive lock for a checkpoint task.

    Args:
        checkpoint_type: ``"manifest"`` or ``"finalize"``. Validated by
            the :data:`phenotypic.sdk_.typing_.CheckpointType` Literal
            alias at type-check time.

    Returns:
        Filename relative to ``<progress>/`` (hidden file, leading ``.``).
    """
    return _CHECKPOINT_LOCK_FILENAME_TEMPLATE.format(checkpoint_type=checkpoint_type)


# ---------------------------------------------------------------------------
# Path-builder helpers — replace inline ``output / "literal"`` constructions
# ---------------------------------------------------------------------------


def phenotypic_cache_dir(output_dir: Path) -> Path:
    """Return ``<output>/.phenotypic/`` — the hidden machine-state root.

    Pure path expression; callers ``mkdir`` when they intend to write.
    """
    return output_dir / DIR_PHENOTYPIC


def progress_dir(output_dir: Path) -> Path:
    """Return ``<output>/.phenotypic/progress/``.

    Pure path expression; callers are responsible for ``mkdir`` when they
    intend to write into it.
    """
    return phenotypic_cache_dir(output_dir) / DIR_PROGRESS


def results_dir(output_dir: Path) -> Path:
    """Return ``<output>/results/``."""
    return output_dir / DIR_RESULTS


def deliverables_dir(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/`` — the user-facing-output folder.

    Pure path expression; callers are responsible for ``mkdir`` when they
    intend to write into it. Writers that go through
    :func:`phenotypic._cli._cli_output_manager._atomic_write` get the
    ``mkdir`` for free (it creates ``target.parent``); direct
    ``write_text``/``write_bytes`` writers must ``mkdir`` explicitly.

    Every artifact helper that previously rooted at ``<output>/`` (master /
    measurements / per-feature split / analysis / dashboard / report /
    pipeline.json / README) now composes from here, so a future relocation
    is a one-line change.
    """
    return output_dir / DIR_DELIVERABLES


def event_log_path(output_dir: Path) -> Path:
    """Return ``<output>/.phenotypic/processing_events.log``."""
    return phenotypic_cache_dir(output_dir) / PROCESSING_EVENTS_LOG


def processing_state_path(output_dir: Path) -> Path:
    """Return ``<output>/.phenotypic/processing_state.json``."""
    return phenotypic_cache_dir(output_dir) / PROCESSING_STATE_JSON


def _legacy_progress_dir(output_dir: Path) -> Path:
    """Pre-migration location: ``<output>/progress/``."""
    return output_dir / DIR_PROGRESS


def _legacy_processing_state_path(output_dir: Path) -> Path:
    """Pre-migration location: ``<output>/processing_state.json``."""
    return output_dir / PROCESSING_STATE_JSON


def resolve_progress_dir(output_dir: Path) -> Path:
    """Return the progress dir that exists, preferring ``.phenotypic/``.

    Read-only helper for resume/discovery so a pre-migration run (progress
    at the output root) is still found. Falls back to the new location when
    neither exists (the default for fresh writes).
    """
    new = progress_dir(output_dir)
    if new.exists():
        return new
    legacy = _legacy_progress_dir(output_dir)
    if legacy.exists():
        return legacy
    return new


def resolve_processing_state_path(output_dir: Path) -> Path:
    """Return the processing-state file that exists, preferring ``.phenotypic/``."""
    new = processing_state_path(output_dir)
    if new.exists():
        return new
    legacy = _legacy_processing_state_path(output_dir)
    if legacy.exists():
        return legacy
    return new


def resolve_manifest_json_path(output_dir: Path) -> Path:
    """Return ``<progress>/manifest.json`` resolving the progress dir for legacy runs."""
    return resolve_progress_dir(output_dir) / MANIFEST_JSON


def resolve_event_log_path(output_dir: Path) -> Path:
    """Return the event log sibling of the resolved progress dir.

    The event log lives beside ``progress/`` (D14): in ``.phenotypic/`` for a
    migrated/new run, at the output root for a not-yet-migrated legacy read.
    Read-only helper for resume/discovery; never mutates the run dir.
    """
    return resolve_progress_dir(output_dir).parent / PROCESSING_EVENTS_LOG


def migrate_legacy_machine_state(output_dir: Path) -> bool:
    """Move a pre-migration run's machine-state into ``.phenotypic/``.

    If legacy machine-state (``progress/``, ``processing_state.json``,
    ``processing_events.log``) is present at the output root, move each artifact
    into the ``.phenotypic/`` cache so the run proceeds coherently against a
    single location. A no-op when no legacy state is present or everything is
    already migrated.

    Robust to interruption and concurrency (the SLURM array case): each artifact
    is moved only when its source still exists and its destination does not, so a
    migration interrupted mid-move *completes* on the next call rather than
    leaving split state; and a lost move race (a concurrent worker moved the
    artifact first) is ignored rather than crashing. Keying per-artifact instead
    of on ``cache.exists()`` is what makes both safe.

    Returns:
        ``True`` if this call moved anything, else ``False``.
    """
    import shutil

    cache = phenotypic_cache_dir(output_dir)
    legacy_progress = _legacy_progress_dir(output_dir)
    legacy_state = _legacy_processing_state_path(output_dir)
    legacy_events = output_dir / PROCESSING_EVENTS_LOG
    if not (legacy_progress.exists() or legacy_state.exists() or legacy_events.exists()):
        return False
    cache.mkdir(parents=True, exist_ok=True)
    moved = False
    for src, dst in (
        (legacy_progress, cache / DIR_PROGRESS),
        (legacy_state, cache / PROCESSING_STATE_JSON),
        (legacy_events, cache / PROCESSING_EVENTS_LOG),
    ):
        if src.exists() and not dst.exists():
            try:
                shutil.move(str(src), str(dst))
                moved = True
            except (FileNotFoundError, shutil.Error):
                # Lost a migration race with a concurrent worker; the winner is
                # moving (or has moved) this artifact — safe to skip.
                pass
    return moved


def clear_machine_state(output_dir: Path) -> bool:
    """Remove **all** of a run's machine-state for a clean ``--restart``.

    Deletes the ``.phenotypic/`` cache (``progress/``, ``processing_state.json``,
    ``processing_events.log``) and any pre-migration root-level machine-state,
    while leaving user-facing output artifacts (``deliverables/``, ``results/``,
    ``qc/``, ``logs/``, …) untouched. This is the difference between ``--restart``
    (re-run the orchestration against clean state, keep outputs) and
    ``--overwrite`` (delete the whole output dir). Clearing the event log here is
    what stops a restart from appending to — and rebuilding its manifest/failure
    records from — the prior run's events.

    Returns:
        ``True`` if any machine-state was removed, else ``False``.
    """
    import shutil

    removed = False
    # Current layout: the hidden cache holds state + event log + progress/.
    cache = phenotypic_cache_dir(output_dir)
    if cache.exists():
        shutil.rmtree(cache)
        removed = True
    # Pre-migration (legacy) root-level machine-state, if a legacy run is restarted.
    legacy_progress = _legacy_progress_dir(output_dir)
    if legacy_progress.exists():
        shutil.rmtree(legacy_progress)
        removed = True
    for legacy_file in (
        _legacy_processing_state_path(output_dir),
        output_dir / PROCESSING_EVENTS_LOG,
    ):
        if legacy_file.exists():
            legacy_file.unlink()
            removed = True
    return removed


def master_measurements_csv_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/master_measurements.csv``."""
    return deliverables_dir(output_dir) / MASTER_MEASUREMENTS_CSV


def master_measurements_parquet_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/master_measurements.parquet``."""
    return deliverables_dir(output_dir) / MASTER_MEASUREMENTS_PARQUET


def measurements_csv_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/measurements.csv`` (post-applied mirror)."""
    return deliverables_dir(output_dir) / MEASUREMENTS_CSV


def measurements_parquet_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/measurements.parquet`` (post-applied mirror)."""
    return deliverables_dir(output_dir) / MEASUREMENTS_PARQUET


def pipeline_json_path(output_dir: Path) -> Path:
    """Return the canonical typed pipeline config path under ``deliverables/``."""
    return deliverables_dir(output_dir) / PIPELINE_JSON


def _legacy_pipeline_json_path(output_dir: Path) -> Path:
    """Return the legacy plain-JSON pipeline config path under ``deliverables/``."""
    return deliverables_dir(output_dir) / _LEGACY_PIPELINE_JSON


def resolve_pipeline_config_path(output_dir: Path) -> Path:
    """Return the best existing pipeline config path for ``output_dir``.

    Resolution prefers the canonical typed path, falls back to legacy
    ``pipeline.json`` when present, and returns the canonical path when neither
    exists so writers naturally create typed config files.
    """
    canonical = pipeline_json_path(output_dir)
    if canonical.exists():
        return canonical
    legacy = _legacy_pipeline_json_path(output_dir)
    if legacy.exists():
        return legacy
    return canonical


def tuning_spec_path(output_dir: Path) -> Path:
    """Return the canonical typed tuning spec path under ``deliverables/``."""
    return deliverables_dir(output_dir) / TUNING_SPEC_JSON


def _legacy_tuning_spec_path(output_dir: Path) -> Path:
    """Return the legacy plain-JSON tuning spec path under ``deliverables/``."""
    return deliverables_dir(output_dir) / _LEGACY_TUNING_SPEC_JSON


def resolve_tuning_spec_path(output_dir: Path) -> Path:
    """Return the best existing tuning spec path for ``output_dir``."""
    canonical = tuning_spec_path(output_dir)
    if canonical.exists():
        return canonical
    legacy = _legacy_tuning_spec_path(output_dir)
    if legacy.exists():
        return legacy
    return canonical


def best_pipeline_path(output_dir: Path) -> Path:
    """Return the canonical typed tuned-winner pipeline path."""
    return deliverables_dir(output_dir) / BEST_PIPELINE_JSON


def _legacy_best_pipeline_path(output_dir: Path) -> Path:
    """Return the legacy plain-JSON tuned-winner pipeline path."""
    return deliverables_dir(output_dir) / _LEGACY_BEST_PIPELINE_JSON


def resolve_best_pipeline_path(output_dir: Path) -> Path:
    """Return the best existing tuned-winner pipeline path for ``output_dir``."""
    canonical = best_pipeline_path(output_dir)
    if canonical.exists():
        return canonical
    legacy = _legacy_best_pipeline_path(output_dir)
    if legacy.exists():
        return legacy
    return canonical


def param_importance_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/param_importance.json`` (the report)."""
    return deliverables_dir(output_dir) / PARAM_IMPORTANCE_JSON


def best_params_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/best_params.json`` (winner params sidecar)."""
    return deliverables_dir(output_dir) / BEST_PARAMS_JSON


def trials_parquet_path(output_dir: Path) -> Path:
    """Return ``<output>/trials.parquet`` (the trial journal; output-dir root)."""
    return Path(output_dir) / TRIALS_PARQUET


def tune_cache_dir(output_dir: Path) -> Path:
    """Return ``<output>/.pht-tune-cache/`` — the tune run's machine-state root.

    The tune-side sibling of :func:`phenotypic_cache_dir`. Pure path expression;
    callers ``mkdir`` when they intend to write.

    Args:
        output_dir: The run output directory.

    Returns:
        ``<output_dir>/.pht-tune-cache/``.
    """
    return Path(output_dir) / DIR_PHT_TUNE_CACHE


def tune_cache_run_marker_path(output_dir: Path) -> Path:
    """Return ``<output>/.pht-tune-cache/run.json`` — the tune-run marker.

    Written at run START (before any deliverable lands) so a live or finished
    tune output is GUI-discoverable. See :data:`RUN_MARKER_JSON`.

    Args:
        output_dir: The run output directory.

    Returns:
        ``<output_dir>/.pht-tune-cache/run.json``.
    """
    return tune_cache_dir(output_dir) / RUN_MARKER_JSON


def tune_cache_study_db_path(output_dir: Path) -> Path:
    """Return ``<output>/.pht-tune-cache/study.db`` (the Optuna study DB).

    The canonical SQLite-WAL storage for the Optuna-backed
    :class:`OptunaStudyStore` when the ``tune`` extra is installed, relocated
    into the hidden tune cache. A legacy run wrote it at the output root; use
    :func:`resolve_study_db_path` to read either location.

    Args:
        output_dir: The run output directory.

    Returns:
        ``<output_dir>/.pht-tune-cache/study.db``.
    """
    return tune_cache_dir(output_dir) / STUDY_DB


def tune_cache_splits_dir(output_dir: Path) -> Path:
    """Return ``<output>/.pht-tune-cache/splits/`` — the held-out split folder.

    Machine state that must survive a fresh-master rewrite and gate resume, so
    it lives in the hidden tune cache, **not** under :func:`deliverables_dir`.
    Pure path expression; callers ``mkdir`` when they intend to write.

    Args:
        output_dir: The run output directory.

    Returns:
        ``<output_dir>/.pht-tune-cache/splits/``.
    """
    return tune_cache_dir(output_dir) / DIR_SPLITS


def tune_cache_split_assignment_path(output_dir: Path) -> Path:
    """Return ``<output>/.pht-tune-cache/splits/split.json`` — the held-out split.

    The persisted calibration / held-out partition (plate names + split kind +
    dataset identity + seed entropy). Read-if-exists-else-derive on resume, so
    a re-run reuses the original partition regardless of the new master seed. A
    legacy run wrote it under ``<output>/splits/``; use
    :func:`resolve_split_assignment_path` to read either location.

    Args:
        output_dir: The run output directory.

    Returns:
        ``<output_dir>/.pht-tune-cache/splits/split.json``.
    """
    return tune_cache_splits_dir(output_dir) / SPLIT_ASSIGNMENT_JSON


def _legacy_study_db_path(output_dir: Path) -> Path:
    """Pre-relocation location: ``<output>/study.db``."""
    return Path(output_dir) / STUDY_DB


def _legacy_split_assignment_path(output_dir: Path) -> Path:
    """Pre-relocation location: ``<output>/splits/split.json``."""
    return Path(output_dir) / DIR_SPLITS / SPLIT_ASSIGNMENT_JSON


def resolve_study_db_path(output_dir: Path) -> Path:
    """Return the study DB that exists, preferring ``.pht-tune-cache/``.

    Read-only resolver: a relocated run keeps ``study.db`` under the hidden tune
    cache; a legacy run kept it at the output root. Falls back to the new
    location when neither exists (so a cold sampler restart from a missing
    ``study.db`` is harmless — no migration is performed).

    Args:
        output_dir: The run output directory.

    Returns:
        The study DB path that exists, else the new cache location.
    """
    new = tune_cache_study_db_path(output_dir)
    if new.exists():
        return new
    legacy = _legacy_study_db_path(output_dir)
    if legacy.exists():
        return legacy
    return new


def resolve_split_assignment_path(output_dir: Path) -> Path:
    """Return the split assignment that exists, preferring ``.pht-tune-cache/``.

    Read-only resolver mirroring :func:`resolve_progress_dir`. The held-out
    split is checked in the hidden tune cache FIRST, THEN at the legacy output
    root — a missing split silently RE-DERIVES a fresh held-out partition on
    resume (a reproducibility / held-out-leak bug), so resume MUST find a
    legacy-root ``split.json``. Falls back to the new location when neither
    exists (the default for a fresh derive-and-write).

    Args:
        output_dir: The run output directory.

    Returns:
        The split-assignment path that exists, else the new cache location.
    """
    new = tune_cache_split_assignment_path(output_dir)
    if new.exists():
        return new
    legacy = _legacy_split_assignment_path(output_dir)
    if legacy.exists():
        return legacy
    return new


def generalization_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/generalization.json`` — the held-out report.

    The winner's generalization verdict (calibration vs held-out score, the gap,
    and the pass/fail margin), a user-facing deliverable. The held-out pass that
    writes it is Phase 4.5 part 2; this helper resolves the canonical location.

    Args:
        output_dir: The run output directory.

    Returns:
        ``<output_dir>/deliverables/generalization.json``.
    """
    return deliverables_dir(output_dir) / GENERALIZATION_JSON


def pareto_dir(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/pareto/`` — the multi-objective sub-folder.

    Holds a multi-objective tune run's Pareto front + per-objective best
    pipelines. A single-objective run never creates it (the back-compat lock).
    """
    return deliverables_dir(output_dir) / DIR_PARETO


def pareto_front_parquet_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/pareto/pareto_front.parquet`` (the front)."""
    return pareto_dir(output_dir) / PARETO_FRONT_PARQUET


def pareto_best_pipeline_path(output_dir: Path, objective: str) -> Path:
    """Return ``deliverables/pareto/best_<objective>.json`` (a per-axis winner).

    The pipeline maximizing the single ``objective`` axis on the Pareto front.
    ``objective`` is the objective name as it appears in ``objectives_json`` (a
    scorer-defined label, e.g. ``"Dice"`` or a composite child handle ``"s0"``).

    Args:
        output_dir: The run directory.
        objective: The objective-axis name (the ``best_<objective>.json`` stem).

    Returns:
        The per-objective best-pipeline path under :func:`pareto_dir`.
    """
    return pareto_dir(output_dir) / _PARETO_BEST_PIPELINE_FILENAME_TEMPLATE.format(
        objective=objective
    )


def pareto_importance_path(output_dir: Path, objective: str) -> Path:
    """Return ``deliverables/pareto/param_importance_<objective>.json``.

    The per-objective RF-permutation importance report (the multi-objective
    sibling of :func:`param_importance_path`). ``objective`` is the objective
    name as it appears in ``objectives_json`` (a scorer-defined label, e.g.
    ``"Dice"`` or a composite child handle ``"s0"``).

    Args:
        output_dir: The run directory.
        objective: The objective-axis name (the filename's ``<objective>`` slot).

    Returns:
        The per-objective importance-report path under :func:`pareto_dir`.
    """
    return pareto_dir(output_dir) / _PARETO_IMPORTANCE_FILENAME_TEMPLATE.format(
        objective=objective
    )


def phenotypic_cache_pipeline_json_path(output_dir: Path) -> Path:
    """Return ``<output>/.phenotypic/pipeline.json`` — the process-only run's
    reproducibility copy. Distinct from :func:`pipeline_json_path`, which roots
    under ``deliverables/`` (process-only writes no deliverables)."""
    return phenotypic_cache_dir(output_dir) / PIPELINE_JSON


def analysis_csv_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/analysis.csv``."""
    return deliverables_dir(output_dir) / ANALYSIS_CSV


def analysis_parquet_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/analysis.parquet``."""
    return deliverables_dir(output_dir) / ANALYSIS_PARQUET


def dashboard_html_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/dashboard.html``."""
    return deliverables_dir(output_dir) / DASHBOARD_HTML


def dataset_results_dir(output_dir: Path, dataset: str) -> Path:
    """Return ``<output>/results/<dataset>/``."""
    return results_dir(output_dir) / dataset


def dataset_measurements_dir(output_dir: Path, dataset: str) -> Path:
    """Return ``<output>/results/<dataset>/measurements/``."""
    return dataset_results_dir(output_dir, dataset) / DIR_MEASUREMENTS


def dataset_hdf_dir(output_dir: Path, dataset: str) -> Path:
    """Return ``<output>/results/<dataset>/hdf/``."""
    return dataset_results_dir(output_dir, dataset) / DIR_HDF


def dataset_overlays_dir(output_dir: Path, dataset: str) -> Path:
    """Return ``<output>/results/<dataset>/overlays/``."""
    return dataset_results_dir(output_dir, dataset) / DIR_OVERLAYS


def measurements_by_feature_dir(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/measurements_by_feature/``."""
    return deliverables_dir(output_dir) / DIR_MEASUREMENTS_BY_FEATURE


def logs_dir(output_dir: Path) -> Path:
    """Return ``<output>/logs/`` (SLURM stdout/stderr + sbatch scripts)."""
    return output_dir / DIR_LOGS


def chunks_dir(progress_dir_: Path) -> Path:
    """Return ``<progress>/chunks/`` for the dashboard chunk parquets."""
    return progress_dir_ / DIR_CHUNKS


def recompile_dir(progress_dir_: Path) -> Path:
    """Return ``<progress>/recompile/``."""
    return progress_dir_ / DIR_RECOMPILE


def recompile_status_dir(progress_dir_: Path) -> Path:
    """Return ``<progress>/recompile/status/``."""
    return recompile_dir(progress_dir_) / DIR_RECOMPILE_STATUS


def task_status_path(output_dir: Path, task_index: int) -> Path:
    """Return ``<progress>/recompile/status/task_<idx>.json``."""
    return recompile_status_dir(progress_dir(output_dir)) / task_status_filename(task_index)


def chunk_parquet_path(progress_dir_: Path, chunk_id: int) -> Path:
    """Return ``<progress>/chunks/chunk_<id:03d>.parquet``."""
    return chunks_dir(progress_dir_) / chunk_parquet_filename(chunk_id)


def checkpoint_lock_path(progress_dir_: Path, checkpoint_type: CheckpointType) -> Path:
    """Return ``<progress>/.{checkpoint_type}_lock``."""
    return progress_dir_ / checkpoint_lock_filename(checkpoint_type)


def job_metadata_path(output_dir: Path) -> Path:
    """Return ``<output>/progress/job_metadata.json``."""
    return progress_dir(output_dir) / JOB_METADATA_JSON


def failures_jsonl_path(output_dir: Path) -> Path:
    """Return ``<output>/progress/failures.jsonl``."""
    return progress_dir(output_dir) / FAILURES_JSONL


def manifest_json_path(output_dir: Path) -> Path:
    """Return ``<output>/progress/manifest.json`` (dashboard manifest)."""
    return progress_dir(output_dir) / MANIFEST_JSON


def chunk_manifest_path(output_dir: Path) -> Path:
    """Return ``<output>/progress/chunk_manifest.json``."""
    return progress_dir(output_dir) / CHUNK_MANIFEST_JSON


def chunk_state_path(output_dir: Path) -> Path:
    """Return ``<output>/progress/chunk_state.json``."""
    return progress_dir(output_dir) / CHUNK_STATE_JSON


def overlay_manifest_path(output_dir: Path) -> Path:
    """Return ``<output>/progress/overlay_manifest.json``."""
    return progress_dir(output_dir) / OVERLAY_MANIFEST_JSON


def analysis_html_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/analysis.html``."""
    return deliverables_dir(output_dir) / ANALYSIS_HTML


def processing_report_html_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/processing_report.html``."""
    return deliverables_dir(output_dir) / PROCESSING_REPORT_HTML


def readme_md_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/README.md``."""
    return deliverables_dir(output_dir) / README_MD


def analysis_full_parquet_path(progress_dir_: Path) -> Path:
    """Return ``<progress>/analysis_full.parquet`` (incremental combined frame).

    Takes a *progress_dir* (not the run output root) since this file lives
    inside ``progress/`` — it's an internal mid-run artifact, not a user-
    facing output.
    """
    return progress_dir_ / ANALYSIS_FULL_PARQUET


def analysis_scatter_json_path(progress_dir_: Path) -> Path:
    """Return ``<progress>/analysis_scatter.json``."""
    return progress_dir_ / ANALYSIS_SCATTER_JSON


def sentinel_resubmitted_path(progress_dir_: Path) -> Path:
    """Return ``<progress>/sentinel_resubmitted`` marker file path."""
    return progress_dir_ / SENTINEL_RESUBMITTED_MARKER


def chunk_lock_path(progress_dir_: Path) -> Path:
    """Return ``<progress>/.chunk_lock``."""
    return progress_dir_ / CHUNK_LOCK


def slurm_scripts_dir(output_dir: Path) -> Path:
    """Return ``<output>/slurm_scripts/``."""
    return output_dir / DIR_SLURM_SCRIPTS


def qc_dir(output_dir: Path) -> Path:
    """Return ``<output>/qc/``.

    Pure path expression; ``run_qc`` is responsible for ``mkdir`` when it
    writes into it (via :func:`_atomic_write`).
    """
    return output_dir / DIR_QC


def qc_summary_parquet_path(output_dir: Path) -> Path:
    """Return ``<output>/qc/qc_summary.parquet``."""
    return qc_dir(output_dir) / QC_SUMMARY_PARQUET


def qc_members_parquet_path(output_dir: Path) -> Path:
    """Return ``<output>/qc/qc_members.parquet``."""
    return qc_dir(output_dir) / QC_MEMBERS_PARQUET


def qc_config_json_path(output_dir: Path) -> Path:
    """Return ``<output>/qc/qc_config.json``."""
    return qc_dir(output_dir) / QC_CONFIG_JSON


def qc_review_state_path(output_dir: Path) -> Path:
    """Return ``<output>/qc/review_state.json`` (GUI-owned review progress)."""
    return qc_dir(output_dir) / QC_REVIEW_STATE_JSON


def errors_dir(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/errors/`` (per-category error parquets)."""
    return deliverables_dir(output_dir) / DIR_ERRORS


def error_category_parquet_path(output_dir: Path, category: str) -> Path:
    """Return ``<output>/deliverables/errors/<category>.parquet``.

    Args:
        output_dir: Run output directory.
        category: A bare, already-sanitized category token (e.g.
            ``"background_noise"``). The caller is responsible for sanitization.
    """
    return errors_dir(output_dir) / f"{category}.parquet"


def error_analysis_parquet_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/error_analysis.parquet``."""
    return deliverables_dir(output_dir) / ERROR_ANALYSIS_PARQUET


def error_analysis_csv_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/error_analysis.csv``."""
    return deliverables_dir(output_dir) / ERROR_ANALYSIS_CSV


def error_analysis_html_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/error_analysis.html``."""
    return deliverables_dir(output_dir) / ERROR_ANALYSIS_HTML


def verified_parquet_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/verified.parquet`` (GUI-written, §9)."""
    return deliverables_dir(output_dir) / VERIFIED_PARQUET


def curation_labels_parquet_path(output_dir: Path) -> Path:
    """Return ``<output>/qc/curation_labels.parquet`` (durable labels store)."""
    return qc_dir(output_dir) / CURATION_LABELS_PARQUET


def custom_categories_json_path(output_dir: Path) -> Path:
    """Return ``<output>/qc/custom_categories.json`` (custom-category registry)."""
    return qc_dir(output_dir) / CUSTOM_CATEGORIES_JSON


# ---------------------------------------------------------------------------
# Loader / reader helpers (consolidate 3 duplicated read sites each)
# ---------------------------------------------------------------------------


def read_run_manifest(output_dir: Path) -> Optional[dict]:
    """Read the run manifest if present, resolving legacy layouts.

    Reads ``<output>/.phenotypic/progress/manifest.json``, falling back to the
    pre-migration ``<output>/progress/manifest.json`` for legacy runs (via
    :func:`resolve_manifest_json_path`). Replaces 4 inline
    ``json.loads(manifest_path.read_text())`` blocks.

    Args:
        output_dir: Run output directory containing ``progress/``.

    Returns:
        Parsed manifest dict, or :data:`None` when the file is missing
        or unparseable (callers can decide whether absence is fatal).
    """
    path = resolve_manifest_json_path(output_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to parse %s; treating as missing", path, exc_info=True)
        return None


def load_master_measurements(output_dir: Path) -> Optional["pl.DataFrame"]:
    """Read ``<output>/master_measurements.csv`` into a polars DataFrame.

    Args:
        output_dir: Run output directory.

    Returns:
        DataFrame, or :data:`None` when the file is missing.
    """
    import polars as pl  # type: ignore[import-not-found]  # lazy

    path = master_measurements_csv_path(output_dir)
    if not path.exists():
        return None
    return pl.read_csv(path)


def resolve_execution_mode(job_meta: Optional[dict]) -> ExecutionMode:
    """Extract :data:`ExecutionMode` from job metadata, defaulting to ``"local"``.

    Replaces a 5-site copy-paste of the
    ``job_meta.get("execution_mode", "local") if job_meta else "local"``
    pattern.

    **Silent coercion:** any value that isn't exactly ``"slurm"`` collapses
    to ``"local"`` — including ``None`` (no metadata file), ``{}`` (no
    key), garbage strings (e.g. ``"validate"``, ``""``), and the literal
    ``None`` value. The function never raises. Callers who need to detect
    an unknown mode and warn / refuse should inspect ``job_meta`` directly
    before calling this helper.

    Args:
        job_meta: Parsed ``progress/job_metadata.json`` content, or
            :data:`None` when the file is absent.

    Returns:
        ``"local"`` or ``"slurm"``.
    """
    if not job_meta:
        return "local"
    raw = job_meta.get(JobMetadataKey.EXECUTION_MODE, "local")
    if raw == "slurm":
        return "slurm"
    return "local"


# ---------------------------------------------------------------------------
# Cross-file JSON contract keys
# ---------------------------------------------------------------------------


class JobMetadataKey:
    """Keys inside ``<output>/progress/job_metadata.json``.

    Writers (CLI execution strategies) and readers (recompile worker,
    sentinel, checkpoint handler, GUI runs registry) must reference
    these constants — never the bare string. Renaming a key here
    should fail fast at every site.
    """

    EXECUTION_MODE: Final[str] = "execution_mode"
    START_TIME: Final[str] = "start_time"
    INPUT_PATH: Final[str] = "input_path"
    METADATA_CSV: Final[str] = "metadata_csv"
    #: Whether the recompile finalizer task should skip QC compute. Set on
    #: the SLURM recompile finalizer task dict alongside ``METADATA_CSV``;
    #: read by ``_cli_recompile_worker._run_post_master_steps``.
    NO_QC: Final[str] = "no_qc"
    SLURM_JOB_IDS: Final[str] = "slurm_job_ids"
    CHUNK_JOB_IDS: Final[str] = "chunk_job_ids"
    CHUNK_SCRIPTS: Final[str] = "chunk_scripts"
    DATASETS: Final[str] = "datasets"
    INCLUDE_DATASET_COLUMN: Final[str] = "include_dataset_column"
    IMAGE_TASK_MAPPING: Final[str] = "image_task_mapping"


class DashboardManifestKey:
    """Keys inside ``<output>/progress/manifest.json`` (dashboard manifest).

    The manifest is built by :func:`_cli._dashboard._manifest_builder.build_manifest`
    and consumed by both the dashboard JS and the GUI run-console's runs
    registry (``_runs_registry.py``). Writers and readers must reference
    these constants rather than spelling the bare string.
    """

    VERSION: Final[str] = "version"
    LAST_UPDATED: Final[str] = "last_updated"
    EXECUTION_MODE: Final[str] = "execution_mode"
    TOTAL_IMAGES: Final[str] = "total_images"
    COMPLETED: Final[str] = "completed"
    FAILED: Final[str] = "failed"
    STARTED: Final[str] = "started"
    PENDING: Final[str] = "pending"
    SUCCESS_RATE: Final[str] = "success_rate"
    IS_COMPLETE: Final[str] = "is_complete"
    START_TIME: Final[str] = "start_time"
    INPUT_PATH: Final[str] = "input_path"
    DATASETS: Final[str] = "datasets"
    FAILURE_CATEGORIES: Final[str] = "failure_categories"
    ANALYSIS_DATA_VERSION: Final[str] = "analysis_data_version"
    SLURM_INFO: Final[str] = "slurm_info"


class DashboardManifestSlurmInfoKey:
    """Keys inside the ``slurm_info`` sub-dict of the dashboard manifest.

    Distinct from :class:`JobMetadataKey`, even when string values overlap —
    these describe the *manifest* contract, not the job-metadata sidecar.
    """

    CHUNK_SCRIPTS: Final[str] = "chunk_scripts"
    TOTAL_CHUNKS: Final[str] = "total_chunks"
    CHUNK_JOB_IDS: Final[str] = "chunk_job_ids"
    ACTIVE_CHUNKS: Final[str] = "active_chunks"
    COMPLETED_CHUNKS: Final[str] = "completed_chunks"
    PENDING_CHUNKS: Final[str] = "pending_chunks"


class ChunkStateKey:
    """Keys inside ``<output>/progress/chunk_state.json``."""

    CHUNKED_FILES: Final[str] = "chunked_files"
    NEXT_CHUNK_ID: Final[str] = "next_chunk_id"


class ProcessingStateKey:
    """Keys inside ``<output>/processing_state.json`` (resume-state file).

    Distinct from :class:`JobMetadataKey` even where string values overlap
    (e.g. ``EXECUTION_MODE``, ``INPUT_PATH``) — these describe the
    `processing_state.json` contract, not the SLURM job metadata sidecar.
    Some values intentionally match across the two contracts so that a
    single field (like ``execution_mode``) can be migrated atomically;
    the ``test_processing_state_keys_match_job_metadata_keys`` regression
    test asserts the overlap.
    """

    VERSION: Final[str] = "version"
    PIPELINE_PATH: Final[str] = "pipeline_path"
    INPUT_PATH: Final[str] = "input_path"
    OUTPUT_DIR: Final[str] = "output_dir"
    TIMESTAMP: Final[str] = "timestamp"
    EXECUTION_MODE: Final[str] = "execution_mode"
    LAST_UPDATED: Final[str] = "last_updated"
    DATASETS: Final[str] = "datasets"
    CONFIG: Final[str] = "config"
    # Per-dataset state-dict sub-keys
    COMPLETED: Final[str] = "completed"
    FAILED: Final[str] = "failed"
    STARTED: Final[str] = "started"
    ERRORS: Final[str] = "errors"
    INITIAL_IMAGES: Final[str] = "initial_images"


class ChunkManifestKey:
    """Keys inside ``<output>/progress/chunk_manifest.json``."""

    CHUNKS: Final[str] = "chunks"
    ROWS: Final[str] = "rows"
    DATASETS: Final[str] = "datasets"
    TOTAL_ROWS: Final[str] = "total_rows"
    NAME: Final[str] = "name"


class HdfAttr:
    """Top-level attribute keys on per-image HDF5 files."""

    PHENOTYPIC_CLASS: Final[str] = "phenotypic_class"


# ---------------------------------------------------------------------------
# Module-path constants for importlib dispatch
# ---------------------------------------------------------------------------


class ModulePath:
    """Importable module paths used in dynamic ``importlib.import_module`` dispatch.

    Spelled out here so a renamed sub-package fails at type-check time
    (consumers reference ``ModulePath.POST`` — a typo there is caught
    by mypy) rather than silently at runtime.
    """

    POST: Final[str] = "phenotypic.post"
    ANALYSIS: Final[str] = "phenotypic.analysis"


# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------


class EnvVar:
    """Environment variable names read or set by the CLI.

    SLURM injects these into batch scripts; the CLI reads them to
    discover its execution context (job id, array task id, …) and
    to find node-local scratch storage.
    """

    SCRATCH: Final[str] = "SCRATCH"
    SLURM_JOB_ID: Final[str] = "SLURM_JOB_ID"
    SLURM_ARRAY_JOB_ID: Final[str] = "SLURM_ARRAY_JOB_ID"
    SLURM_ARRAY_TASK_ID: Final[str] = "SLURM_ARRAY_TASK_ID"
    SLURM_ARRAY_TASK_COUNT: Final[str] = "SLURM_ARRAY_TASK_COUNT"
    SLURM_CPUS_PER_TASK: Final[str] = "SLURM_CPUS_PER_TASK"
    SLURM_MEM_PER_NODE: Final[str] = "SLURM_MEM_PER_NODE"


# ---------------------------------------------------------------------------
# HDF image-class reader (eliminates 3-site duplication)
# ---------------------------------------------------------------------------


def load_image_from_hdf(
    hdf_path: Path,
    *,
    fallback: ImageTypeName = "Image",
) -> "_Image | _GridImage":
    """Open an HDF5, read its ``phenotypic_class`` attr, dispatch to the right Image class.

    Replaces the 3 ad-hoc ``h5py.File(...) → fh.attrs.get('phenotypic_class', 'Image')
    → GridImage if cls_attr == 'GridImage' else Image`` patterns in
    :mod:`_cli_recompile_worker`, :mod:`_cli_execution_strategies`, and
    :mod:`phenotypicCLI`.

    Args:
        hdf_path: Path to a per-image HDF5 file.
        fallback: Image class name to use when the HDF lacks the
            ``phenotypic_class`` attribute (legacy files). Type-checked
            (statically) against :data:`ImageTypeName` — there is no
            runtime validation; the only effect of an unrecognized
            string is that the dispatch falls through to :class:`Image`.

    Returns:
        An :class:`Image` or :class:`GridImage` instance loaded from the HDF.
    """
    import h5py  # type: ignore[import-untyped]

    from phenotypic import GridImage, Image  # lazy: avoids circular import at module load
    from phenotypic.sdk_.constants_ import IMAGE_TYPES

    with h5py.File(hdf_path, "r") as fh:
        cls_attr = fh.attrs.get(HdfAttr.PHENOTYPIC_CLASS, fallback)
    if isinstance(cls_attr, bytes):
        cls_attr = cls_attr.decode("utf-8", errors="replace")
    image_cls = GridImage if cls_attr == IMAGE_TYPES.GRID.value else Image
    return image_cls.load_hdf5(hdf_path)
