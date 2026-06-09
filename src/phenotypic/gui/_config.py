"""Shared GUI constants: launcher defaults, mount paths, server-config keys.

Single source of truth for values that previously lived as string literals
across multiple tools (shell, builder, results viewer, run console).
Adding a new constant here is preferred over re-spelling a magic string
in a callback or layout module.

Module split rationale
----------------------
This module is intentionally narrow:

    * **Python identifiers only** (host strings, port ints, config-dict
      keys, mount-point prefixes, sandbox subdirectory names).
    * **No Dash / Flask / dash-bootstrap-components imports** at module
      top level so `_config` stays cheap to import from anywhere
      (callbacks, blueprints, tests).

Design tokens (colors, type scale, radius, shadow, ease) live in the
sibling :mod:`phenotypic.gui._design` module because they have a CSS
injection helper (:func:`_design.inject_design_tokens`) that this
module shouldn't pull in.

The CLI ↔ GUI shared output filenames (``MASTER_MEASUREMENTS_PARQUET``,
``MEASUREMENTS_CSV``, ``MEASUREMENTS_PARQUET``, ``ANALYSIS_CSV``,
``ANALYSIS_PARQUET``, ``PIPELINE_JSON``, ``RESULTS_DIRNAME``,
``PROGRESS_DIRNAME``, ``DELIVERABLES_DIRNAME``, ``DASHBOARD_FILENAME``)
are re-exports of canonical constants in :mod:`phenotypic.tools_._io_constants`.
The single source of truth is one level up; this module re-exports for
ergonomic GUI imports. New filenames written by the CLI should be added
to ``_io_constants.py``, not here.

These are bare *filenames*, not paths. The user-facing run artifacts
(``master_measurements.*``, ``measurements.*``, ``measurements_by_feature/``,
``analysis.*``, ``dashboard.html``, ``analysis.html``,
``processing_report.html``, ``README.md``, ``pipeline.json``) now resolve
under ``<output>/deliverables/`` — ``DELIVERABLES_DIRNAME`` (= ``"deliverables"``,
backed by ``DIR_DELIVERABLES`` in :mod:`phenotypic.tools_`). Join them via the
``phenotypic.tools_`` path helpers (``deliverables_dir(output)``,
``master_measurements_parquet_path(output)``, …). ``RESULTS_DIRNAME`` and
``QC_DIRNAME`` are *not* deliverables and stay at the output-dir root; the
machine-state sidecars ``PROGRESS_DIRNAME`` (``progress/``) and
``processing_state.json`` now live under the hidden ``<output>/.phenotypic/``
cache (``PHENOTYPIC_CACHE_DIRNAME``), resolved for legacy runs via
``resolve_manifest_json_path``.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal

from phenotypic.tools_ import (
    ANALYSIS_CSV,
    ANALYSIS_PARQUET,
    DASHBOARD_HTML,
    DIR_DELIVERABLES,
    DIR_INSPECT,
    DIR_MEASUREMENTS,
    DIR_OVERLAYS,
    DIR_PHENOTYPIC,
    DIR_PROGRESS,
    DIR_QC,
    DIR_RESULTS,
    JOB_METADATA_JSON,
    MANIFEST_JSON,
    MASTER_MEASUREMENTS_CSV,
    MASTER_MEASUREMENTS_PARQUET,
    MEASUREMENTS_CSV,
    MEASUREMENTS_PARQUET,
    PIPELINE_JSON,
    QC_CONFIG_JSON,
    QC_MEMBERS_PARQUET,
    QC_REVIEW_STATE_JSON,
    QC_SUMMARY_PARQUET,
    STDOUT_LOG,
)

__all__ = [
    # Launcher defaults
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "LOG_FORMAT",
    # Mount-point prefixes
    "MOUNT_HOME",
    "MOUNT_BUILDER",
    "MOUNT_VIEWER",
    "MOUNT_RUN",
    "MOUNT_ANALYSIS",
    "MOUNT_TUNE",
    "SANDBOX_API_PREFIX",
    "RUNS_BLUEPRINT_PREFIX",
    # Flask app.server.config keys
    "CFG_URL_PREFIX",
    "CFG_OPERATION_REGISTRY",
    "CFG_RUN_REGISTRY",
    "CFG_RUNNER",
    "CFG_IMAGE_ROOT",
    "CFG_SANDBOX_ROOT",
    "CFG_OUTPUT_ROOT",
    "CFG_FILTERED_STATE",
    "CFG_RECIPE_STATE",
    "CFG_MEASUREMENT_SCHEMA",
    "CFG_QC_RECIPE",
    "CFG_QC_INSTANCES_CACHE",
    "CFG_QC_AUGMENTED_FRAME",
    "CFG_QC_PIPELINE",
    # Sandbox subdirectories
    "SANDBOX_GUI_DIRNAME",
    "SANDBOX_PRESETS_SUBDIR",
    "SANDBOX_TUNE_PRESETS_SUBDIR",
    "SANDBOX_BUILDER_TILES_SUBDIR",
    "RUN_LOG_DIRNAME",
    "VIEWER_CACHE_DIRNAME",
    # Output filenames (CLI ↔ GUI shared layout) — re-exported from phenotypic.tools_
    "MASTER_MEASUREMENTS_CSV",
    "MASTER_MEASUREMENTS_PARQUET",
    "MEASUREMENTS_CSV",
    "MEASUREMENTS_PARQUET",
    "ANALYSIS_CSV",
    "ANALYSIS_PARQUET",
    "PIPELINE_JSON",
    "JOB_METADATA_JSON",
    "MANIFEST_JSON",
    "STDOUT_LOG",
    "QC_SUMMARY_PARQUET",
    "QC_MEMBERS_PARQUET",
    "QC_CONFIG_JSON",
    "QC_REVIEW_STATE_JSON",
    # Directory names — re-exported from phenotypic.tools_
    "RESULTS_DIRNAME",
    "PROGRESS_DIRNAME",
    "QC_DIRNAME",
    "DELIVERABLES_DIRNAME",
    "DIR_INSPECT",
    "DIR_MEASUREMENTS",
    "DIR_OVERLAYS",
    # Dashboard filename — re-exported from phenotypic.tools_
    "DASHBOARD_FILENAME",
    # URL constants
    "SANDBOX_API_VIEWER_OUTPUT_ROOT",
    "BUILDER_TILES_PREFIX",
    "VIEWER_TILES_PREFIX",
    "COLONY_CROPS_URL_SEGMENT",
    "QC_CROPS_URL_SEGMENT",
    # Closed value-set aliases
    "ChannelName",
    # Tile-spotlight dim strength (shared by both toolbars + the crop route)
    "TILE_DIM_DEFAULT",
    "TILE_DIM_STEP",
    "TILE_DIM_MIN",
    "TILE_DIM_MAX",
    "step_dim_alpha",
    "stepped_alpha_from_trigger",
    # Tunables
    "DEFAULT_IDLE_RELEASE_SECONDS",
    "RSS_INTERVAL_MS",
    # Branding
    "TITLE_HUB",
    "TITLE_BUILDER",
    "TITLE_VIEWER",
    "TITLE_RUN",
    "TITLE_ANALYSIS",
    "TITLE_TUNE",
    "SSH_TUNNEL_HINT",
    # Thread name prefix
    "THREAD_NAME_PREFIX",
    # Helpers
    "add_launcher_args",
    "configure_launcher_logging",
    "print_launcher_banner",
    "tune_presets_dir",
]

# ---------------------------------------------------------------------------
# Launcher defaults (CLI + programmatic)
# ---------------------------------------------------------------------------

DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 8050
LOG_FORMAT: str = "%(asctime)s %(levelname)s %(name)s %(message)s"

# ---------------------------------------------------------------------------
# Mount-point prefixes (shell DispatcherMiddleware + tab nav)
# ---------------------------------------------------------------------------

MOUNT_HOME: str = "/"
MOUNT_BUILDER: str = "/builder/"
MOUNT_VIEWER: str = "/results/"
MOUNT_RUN: str = "/run/"
MOUNT_ANALYSIS: str = "/analysis/"
MOUNT_TUNE: str = "/tune/"

#: Flask blueprint prefix for the sandbox JSON API (sidebar tree, capability
#: probe, viewer hand-off, etc.). Mounted on the shell's Flask server in
#: :func:`phenotypic.gui.shell._app._build_shell_dash_app`.
SANDBOX_API_PREFIX: str = "/sandbox/api"

#: Flask blueprint prefix for run-console iframe + dashboard polling. The
#: Recent Runs panel iframes `<RUNS_BLUEPRINT_PREFIX>/<rel_path>/dashboard.html`.
RUNS_BLUEPRINT_PREFIX: str = "/runs"

# ---------------------------------------------------------------------------
# Flask app.server.config keys
# ---------------------------------------------------------------------------

#: ``app.server.config[CFG_URL_PREFIX]`` — the mount-point prefix this
#: Dash app was built with. Read by callbacks that need to construct
#: hub-aware URLs at request time (colony-grid crops, JS shims).
CFG_URL_PREFIX: str = "pheno_url_prefix"

#: ``app.server.config[CFG_OPERATION_REGISTRY]`` — the builder's
#: :class:`phenotypic.gui.OperationRegistry` handle (catalog of
#: pipeline operations + parameter metadata that drives the palette).
#: Lives on the builder Dash app's Flask server only.
CFG_OPERATION_REGISTRY: str = "pheno_operation_registry"

#: ``app.server.config[CFG_RUN_REGISTRY]`` — the
#: :class:`phenotypic.gui.shell._runs_registry.RunRegistry` handle
#: tracking started pipeline runs (run_id, status, log path). Stored
#: on BOTH the run-console Dash app's Flask server and the shell's
#: Flask server (they share the same instance).
CFG_RUN_REGISTRY: str = "pheno_run_registry"

#: ``app.server.config[CFG_RUNNER]`` — process-wide :class:`LocalRunner`
#: shared by the shell ``/runs/`` blueprint and the run-console UI.
CFG_RUNNER: str = "pheno_runner"

#: ``app.server.config[CFG_IMAGE_ROOT]`` — directory rooting the builder's
#: in-app file picker (or :data:`None` to disable the tree).
CFG_IMAGE_ROOT: str = "pheno_image_root"

#: ``app.server.config[CFG_SANDBOX_ROOT]`` — string path to the frozen
#: sandbox root. Used by run-console form widgets that constrain their
#: pickers to the sandbox.
CFG_SANDBOX_ROOT: str = "pheno_sandbox_root"

#: ``app.server.config[CFG_OUTPUT_ROOT]`` — :class:`OutputRoot` handle on
#: a CLI output directory. Set when the results viewer is in loaded mode.
CFG_OUTPUT_ROOT: str = "output_root"

#: ``app.server.config[CFG_FILTERED_STATE]`` —
#: :class:`FilteredMeasurements` for the loaded results viewer.
CFG_FILTERED_STATE: str = "filtered_state"

#: ``app.server.config`` key holding the analysis sub-app's
#: :class:`~phenotypic.gui.analysis._recipe_state.RecipeState` —
#: a wrapper around the canonical ``<output>/pipeline.json`` that
#: provides atomic save + mtime-staleness detection.
CFG_RECIPE_STATE: str = "recipe_state"

#: ``app.server.config`` key holding the analysis sub-app's
#: :class:`~phenotypic.gui._schema_cache.MeasurementSchema` —
#: a lazy mtime-keyed cache of column lists from
#: ``measurements.parquet`` / ``master_measurements.parquet`` (with CSV
#: fallback). Drives the column-aware dropdowns on filter / model
#: section forms.
CFG_MEASUREMENT_SCHEMA: str = "pheno_measurement_schema"

#: ``app.server.config[CFG_QC_RECIPE]`` — :class:`QcRecipe` instance
#: for the active output directory. Loaded at create_app() boot and
#: mutated by QC tab callbacks (add/remove/update). Spec lines 751-759.
CFG_QC_RECIPE: str = "pheno_qc_recipe"

#: ``app.server.config[CFG_QC_INSTANCES_CACHE]`` — a single dict
#: ``{revision: list[QualityCheck]}`` invalidated on every recipe-revision
#: change (read-then-discard, not unbounded). Spec lines 753-755.
CFG_QC_INSTANCES_CACHE: str = "pheno_qc_instances"

#: ``app.server.config[CFG_QC_AUGMENTED_FRAME]`` — latest merged
#: filtered + QC-columns frame consumed by the Heatmap tab. Single
#: value, overwritten on every QC card-body refresh; sized cap-at-one
#: enforced. Spec lines 756-759.
CFG_QC_AUGMENTED_FRAME: str = "pheno_qc_augmented_frame"

#: ``app.server.config[CFG_QC_PIPELINE]`` — the
#: :class:`~phenotypic._core._image_pipeline.ImagePipeline` deserialized
#: from the active output root's ``pipeline.json`` at boot. The QC Review
#: tab's per-group recompute hands this to
#: :func:`phenotypic.qc._runner.run_qc` so the in-session recompute uses
#: exactly the same checks the CLI persisted. ``None`` when no
#: ``pipeline.json`` exists (or it failed to load) — recompute degrades to
#: a no-op in that case. Spec §D.5.
CFG_QC_PIPELINE: str = "pheno_qc_pipeline"

# ---------------------------------------------------------------------------
# Sandbox subdirectories
# ---------------------------------------------------------------------------

#: Hidden directory inside the sandbox root that holds GUI-managed state
#: (presets, builder tile caches, etc.). Always created lazily.
SANDBOX_GUI_DIRNAME: str = ".phenotypic-gui"

#: Subdirectory of :data:`SANDBOX_GUI_DIRNAME` holding saved run-console
#: presets (one ``<name>.json`` per preset).
SANDBOX_PRESETS_SUBDIR: str = "presets"

#: Subdirectory of ``.phenotypic-gui/presets`` holding saved tuning specs.
SANDBOX_TUNE_PRESETS_SUBDIR: str = "tune"

#: Subdirectory of :data:`SANDBOX_GUI_DIRNAME` holding builder DZI tile
#: caches per loaded image.
SANDBOX_BUILDER_TILES_SUBDIR: str = "builder_tiles"


def tune_presets_dir(sandbox_root: Path) -> Path:
    """Return ``<sandbox>/.phenotypic-gui/presets/tune``."""
    return (
        Path(sandbox_root)
        / SANDBOX_GUI_DIRNAME
        / SANDBOX_PRESETS_SUBDIR
        / SANDBOX_TUNE_PRESETS_SUBDIR
    )

#: Hidden directory inside a run's *output* directory (NOT the sandbox)
#: holding ``stdout.log`` and other on-disk run artifacts.
RUN_LOG_DIRNAME: str = ".gui_log"

#: Hidden directory inside the results viewer's *output* directory
#: holding cached DZI tiles.
VIEWER_CACHE_DIRNAME: str = ".viewer_cache"

# ---------------------------------------------------------------------------
# Output filenames (CLI ↔ GUI shared layout) — re-exported from phenotypic.tools_
# ---------------------------------------------------------------------------
# These names are canonical in phenotypic.tools_._io_constants and re-exported
# here so existing GUI imports (``from phenotypic.gui._config import MEASUREMENTS_CSV``)
# keep working with zero downstream churn. Do NOT redefine these as inline literals.
#
# Available re-exports (imported at module top):
#   MASTER_MEASUREMENTS_CSV, MASTER_MEASUREMENTS_PARQUET,
#   MEASUREMENTS_CSV, MEASUREMENTS_PARQUET,
#   ANALYSIS_CSV, ANALYSIS_PARQUET,
#   PIPELINE_JSON, JOB_METADATA_JSON, MANIFEST_JSON, STDOUT_LOG,
#   QC_SUMMARY_PARQUET, QC_MEMBERS_PARQUET, QC_CONFIG_JSON,
#   QC_REVIEW_STATE_JSON,
#   DIR_RESULTS, DIR_PROGRESS, DIR_QC, DIR_DELIVERABLES, DASHBOARD_HTML
#
# These are filenames/dirnames only. The user-facing artifacts
# (master/measurements/analysis frames, the per-feature splits, the
# generated HTML reports, README, and pipeline.json) now resolve under
# ``<output>/deliverables/`` (DELIVERABLES_DIRNAME / DIR_DELIVERABLES) via the
# phenotypic.tools_ path helpers (deliverables_dir, master_measurements_parquet_path,
# …). The per-image results dir and the qc dir stay at the output-dir root,
# while machine-state (progress dir, processing_state.json, processing_events.log)
# now lives under the hidden cache dir ``<output>/.phenotypic/``
# (PHENOTYPIC_CACHE_DIRNAME / DIR_PHENOTYPIC); resolve it via the
# phenotypic.tools_ helpers (progress_dir, resolve_manifest_json_path, …).
#
# ---------------------------------------------------------------------------
# GUI-only convenience aliases for re-exported CLI artifact names
# ---------------------------------------------------------------------------

#: ``results`` — the CLI output ``results/`` directory name.
RESULTS_DIRNAME: str = DIR_RESULTS

#: ``progress`` — the CLI output ``progress/`` directory name. Now nested
#: under the hidden machine-state cache (see PHENOTYPIC_CACHE_DIRNAME); resolve
#: full paths via the ``phenotypic.tools_`` helpers rather than joining this.
PROGRESS_DIRNAME: str = DIR_PROGRESS

#: ``.phenotypic`` — the hidden machine-state cache dir holding the run's
#: progress/, processing_state.json, and processing_events.log. Distinct from
#: the GUI's ``.phenotypic-gui`` sandbox dir (SANDBOX_GUI_DIRNAME).
PHENOTYPIC_CACHE_DIRNAME: str = DIR_PHENOTYPIC

#: ``qc`` — the CLI output ``qc/`` artifact directory name.
QC_DIRNAME: str = DIR_QC

#: ``deliverables`` — the CLI output ``deliverables/`` directory holding all
#: user-facing run outputs (master/measurements/analysis frames, the
#: per-feature splits, the generated dashboard/analysis/report HTML, README,
#: and the canonical pipeline.json). The shell classifier's CLI-output and
#: dashboard markers and the run console's run-file URLs compose through this.
DELIVERABLES_DIRNAME: str = DIR_DELIVERABLES

#: ``dashboard.html`` — the generated dashboard artifact filename.
DASHBOARD_FILENAME: str = DASHBOARD_HTML

# ---------------------------------------------------------------------------
# URL constants
# ---------------------------------------------------------------------------

#: Full Flask route for the sandbox API viewer output-root handoff endpoint.
SANDBOX_API_VIEWER_OUTPUT_ROOT: str = f"{SANDBOX_API_PREFIX}/viewer/output-root"

#: URL prefix where the builder's DZI tile blueprint mounts on the Flask
#: app — the path the Flask server sees AFTER the hub
#: :class:`DispatcherMiddleware` strips the mount prefix. The browser-
#: facing URL is ``<requests_pathname_prefix>tiles/...``; standalone
#: launches collapse ``requests_pathname_prefix`` to ``/``, hub mode
#: prepends ``/builder/``. Mirrors :data:`VIEWER_TILES_PREFIX` so both
#: tile blueprints follow the same routing convention.
BUILDER_TILES_PREFIX: str = "/tiles"

#: URL prefix for the results-viewer's DZI tile blueprint. Distinct from
#: ``BUILDER_TILES_PREFIX`` because the viewer's tile cache and route
#: namespace are scoped to the results sub-app.
VIEWER_TILES_PREFIX: str = "/tiles"

#: URL path segment used for per-colony crop images.
COLONY_CROPS_URL_SEGMENT: str = "crops"

#: URL path segment used for the QC Review tab's colony crops. Distinct
#: from :data:`COLONY_CROPS_URL_SEGMENT` so the two crop blueprints mount
#: under separate names on the same Flask server (see
#: :func:`phenotypic.gui._shared.tiles.register_crop_route`). Both routes
#: serve identical centered PNGs; the namespaces are kept apart only so
#: the blueprint registrations never collide.
QC_CROPS_URL_SEGMENT: str = "qc-crops"

# ---------------------------------------------------------------------------
# Closed value-set aliases
# ---------------------------------------------------------------------------

#: Image channel names supported by the builder inspector previews.
ChannelName = Literal["rgb", "gray", "detect_mat", "objmap"]

# ---------------------------------------------------------------------------
# Tile-spotlight dim strength (shared numeric policy)
# ---------------------------------------------------------------------------
#
# Single source of truth for the colony-crop "spotlight" effect: the crop
# route fades each tile's surroundings toward black by ``dim_alpha`` while
# keeping the target colony's bbox at full opacity. The same bounds drive
# the route-side clamp on ``?dim=`` and the UI ``−``/``+`` steppers in both
# the colony-view and QC-review toolbars, so they can never disagree.

#: Default spotlight strength when the viewer first loads — the effect is
#: on by default. ``0.0`` would restore today's full-context crop.
TILE_DIM_DEFAULT: float = 0.6

#: Increment per ``−``/``+`` stepper click.
TILE_DIM_STEP: float = 0.05

#: Lowest allowed spotlight strength. ``0.0`` disables the effect entirely
#: (full-context crop, identical to the pre-feature output).
TILE_DIM_MIN: float = 0.0

#: Highest allowed spotlight strength. Capped below ``1.0`` so the
#: surroundings never go fully black — a faint context cue always remains.
TILE_DIM_MAX: float = 0.9


def step_dim_alpha(current: float, direction: int) -> float:
    """Step the tile-spotlight strength one click and clamp it to range.

    Pure arithmetic shared by both toolbars' ``−``/``+`` callbacks so the
    stepping logic stays unit-testable without Dash. The result is rounded
    to two decimal places to avoid binary-float drift accumulating across
    repeated clicks (e.g. ``0.1 + 0.05 + 0.05`` landing on
    ``0.20000000000000001``).

    Args:
        current: The strength before the click, typically read from the
            shared ``STORE_TILE_DIM_ALPHA`` store.
        direction: ``+1`` for the ``+`` button, ``-1`` for the ``−``
            button.

    Returns:
        ``current + direction * TILE_DIM_STEP`` clamped to
        ``[TILE_DIM_MIN, TILE_DIM_MAX]`` and rounded to two decimals.
    """
    stepped = current + direction * TILE_DIM_STEP
    clamped = min(TILE_DIM_MAX, max(TILE_DIM_MIN, stepped))
    return round(clamped, 2)


def stepped_alpha_from_trigger(
    triggered_id: object,
    current: float | None,
    *,
    plus_id: str,
    minus_id: str,
) -> float:
    """Resolve a ``−``/``+`` stepper click into the next clamped strength.

    The thin, **Dash-free** decision shared by both toolbars' stepper
    callbacks (so the Dash callback body stays a one-line adapter over
    this unit-testable helper, per the GUI's inline-closure-500 gotcha).
    Maps the firing button id to a direction and defers the arithmetic to
    :func:`step_dim_alpha`.

    Args:
        triggered_id: ``dash.ctx.triggered_id`` — the id of the button
            that fired (``plus_id`` or ``minus_id``). An unrecognised /
            ``None`` value (e.g. an initial-mount echo) is treated as a
            ``+`` so the strength never silently jumps backwards.
        current: The strength before the click (the store value); ``None``
            falls back to :data:`TILE_DIM_DEFAULT`.
        plus_id: Component id of the ``+`` (step-up) button.
        minus_id: Component id of the ``−`` (step-down) button.

    Returns:
        The next strength, clamped to ``[TILE_DIM_MIN, TILE_DIM_MAX]``.
    """
    base = TILE_DIM_DEFAULT if current is None else float(current)
    if triggered_id == minus_id:
        direction = -1
    elif triggered_id == plus_id:
        direction = 1
    else:
        # Unrecognised / initial-mount echo: step up rather than down so
        # the strength never silently jumps backwards.
        direction = 1
    return step_dim_alpha(base, direction)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

#: How long (seconds) the viewer :class:`ToolSession` may go idle before
#: the daemon thread releases its in-memory state. Defaults to 15 minutes
#: per ``GUI_SPEC_V1.md``. Tests pass an explicit override.
DEFAULT_IDLE_RELEASE_SECONDS: float = 15 * 60.0

#: How often (milliseconds) the top-bar RSS readout polls
#: ``psutil.Process().memory_info().rss``.
RSS_INTERVAL_MS: int = 5_000

# ---------------------------------------------------------------------------
# Branding strings
# ---------------------------------------------------------------------------

TITLE_HUB: str = "PhenoTypic GUI"
TITLE_BUILDER: str = "PhenoTypic Pipeline Builder"
TITLE_VIEWER: str = "PhenoTypic Results Viewer"
TITLE_RUN: str = "PhenoTypic Run Console"
TITLE_ANALYSIS: str = "PhenoTypic Analysis"
TITLE_TUNE: str = "PhenoTypic Tune Co-Pilot"

#: One-line SSH-tunnel hint reused by every launcher banner, argparse
#: epilogue, and help-modal body. Constructed from :data:`DEFAULT_PORT`
#: so a future port change propagates everywhere.
SSH_TUNNEL_HINT: str = (
    f"ssh -L {DEFAULT_PORT}:localhost:{DEFAULT_PORT} user@cluster"
)

#: Shared prefix for ``threading.Thread`` / ``ThreadPoolExecutor`` names
#: spawned by GUI callbacks. Suffixes like ``"-slurm"`` or
#: ``"-idle-release"`` distinguish concrete uses.
THREAD_NAME_PREFIX: str = "phenotypic-gui"

# ---------------------------------------------------------------------------
# Launcher helpers
# ---------------------------------------------------------------------------

def add_launcher_args(
    parser: argparse.ArgumentParser,
    *,
    include_debug: bool = True,
) -> None:
    """Append the shared ``--host``, ``--port``, ``--debug`` flags.

    Standalone debug launchers (builder, results viewer, run console) and
    the unified hub launcher all expose the same trio. Funnelling through
    one helper keeps default values + help text in lock-step.

    Args:
        parser: An :class:`argparse.ArgumentParser` to mutate.
        include_debug: When :data:`False`, omit ``--debug`` (some entry
            points wire it separately or not at all).
    """
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=(
            f"Interface to bind. Default {DEFAULT_HOST} keeps the server "
            "loopback-only — pair with SSH port forwarding for remote "
            "access. Use 0.0.0.0 to expose on the network (not "
            "recommended without authentication)."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"TCP port to bind. Default {DEFAULT_PORT}.",
    )
    if include_debug:
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Run Dash in debug mode (auto-reload, verbose tracebacks).",
        )


def configure_launcher_logging(*, debug: bool) -> None:
    """Apply the shared root-logger configuration used by every launcher.

    Args:
        debug: When :data:`True`, set the root level to ``DEBUG``;
            otherwise ``INFO``.
    """
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=LOG_FORMAT,
    )


def print_launcher_banner(
    *,
    title: str,
    host: str,
    port: int,
    root: Path,
    extra_lines: tuple[str, ...] = (),
) -> None:
    """Print a friendly startup banner with SSH-tunnel hint.

    Replaces the per-launcher ``_print_banner`` helpers in
    ``shell/_launcher.py`` and ``results_viewer/__main__.py``. Standalone
    builder + run-console launchers can opt-in to the shared format by
    calling this from their ``main``.

    Args:
        title: Human-readable app title (one of :data:`TITLE_*`).
        host: Bound interface (echoed back verbatim).
        port: Bound TCP port.
        root: Resolved sandbox / image / output root, surfaced verbatim
            so the user can confirm what the launcher pointed at.
        extra_lines: Optional additional banner lines (e.g. a cache-nuke
            hint specific to the viewer). Each line is printed with the
            same two-space indent as the standard rows.
    """
    print()
    print(title)
    print(f"  root  : {root}")
    print(f"  url   : http://{host}:{port}/")
    print()
    print(f"  SSH tunnel from local: ssh -L {port}:localhost:{port} <cluster>")
    for line in extra_lines:
        print(f"  {line}")
    print()
