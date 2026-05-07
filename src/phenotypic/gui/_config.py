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
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

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
    # Sandbox subdirectories
    "SANDBOX_GUI_DIRNAME",
    "SANDBOX_PRESETS_SUBDIR",
    "SANDBOX_BUILDER_TILES_SUBDIR",
    "RUN_LOG_DIRNAME",
    "VIEWER_CACHE_DIRNAME",
    # Output filenames (CLI ↔ GUI shared layout)
    "MASTER_MEASUREMENTS_PARQUET",
    "MEASUREMENTS_CSV",
    "MEASUREMENTS_PARQUET",
    "ANALYSIS_CSV",
    "ANALYSIS_PARQUET",
    "PIPELINE_JSON",
    # Tunables
    "DEFAULT_IDLE_RELEASE_SECONDS",
    "RSS_INTERVAL_MS",
    # Branding
    "TITLE_HUB",
    "TITLE_BUILDER",
    "TITLE_VIEWER",
    "TITLE_RUN",
    "TITLE_ANALYSIS",
    "SSH_TUNNEL_HINT",
    # Thread name prefix
    "THREAD_NAME_PREFIX",
    # Helpers
    "add_launcher_args",
    "configure_launcher_logging",
    "print_launcher_banner",
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

# ---------------------------------------------------------------------------
# Sandbox subdirectories
# ---------------------------------------------------------------------------

#: Hidden directory inside the sandbox root that holds GUI-managed state
#: (presets, builder tile caches, etc.). Always created lazily.
SANDBOX_GUI_DIRNAME: str = ".phenotypic-gui"

#: Subdirectory of :data:`SANDBOX_GUI_DIRNAME` holding saved run-console
#: presets (one ``<name>.json`` per preset).
SANDBOX_PRESETS_SUBDIR: str = "presets"

#: Subdirectory of :data:`SANDBOX_GUI_DIRNAME` holding builder DZI tile
#: caches per loaded image.
SANDBOX_BUILDER_TILES_SUBDIR: str = "builder_tiles"

#: Hidden directory inside a run's *output* directory (NOT the sandbox)
#: holding ``stdout.log`` and other on-disk run artifacts.
RUN_LOG_DIRNAME: str = ".gui_log"

#: Hidden directory inside the results viewer's *output* directory
#: holding cached DZI tiles.
VIEWER_CACHE_DIRNAME: str = ".viewer_cache"

# ---------------------------------------------------------------------------
# Output filenames (CLI ↔ GUI shared layout)
# ---------------------------------------------------------------------------

#: Aggregate parquet emitted by the CLI finalizer
#: (:func:`phenotypic._cli._cli_output_manager.aggregate_measurements`).
#: Read-only from the GUI's perspective — the results viewer treats this
#: as the immutable source of truth.
MASTER_MEASUREMENTS_PARQUET: str = "master_measurements.parquet"

#: Editable curated CSV mirror seeded by the CLI as a copy of
#: ``master_measurements.csv``. The results viewer rewrites this file in
#: place when the user removes/restores colonies. Re-running the CLI
#: (forward, ``--measure``, ``--recompile``) overwrites it with a fresh
#: full copy, intentionally wiping prior curation.
MEASUREMENTS_CSV: str = "measurements.csv"

#: Editable curated parquet companion to :data:`MEASUREMENTS_CSV`. The
#: parquet is the GUI's source of truth at boot (CSV is the human-readable
#: mirror); both are written atomically together.
MEASUREMENTS_PARQUET: str = "measurements.parquet"

#: Output filename of the model-fit summary written by the CLI when the
#: pipeline has a ``model`` configured (and re-emitted by the analysis
#: GUI's "Run analysis" button). Human-readable mirror of
#: :data:`ANALYSIS_PARQUET`.
ANALYSIS_CSV: str = "analysis.csv"

#: Parquet companion of :data:`ANALYSIS_CSV` — primary downstream
#: artifact since it preserves dtypes the CSV cannot.
ANALYSIS_PARQUET: str = "analysis.parquet"

#: Canonical pipeline-spec filename written into the output root by the
#: CLI (and rewritten by the analysis GUI on every recipe edit). Captures
#: operations, measurements, post, filters, and model — i.e. the whole
#: reproducibility surface.
PIPELINE_JSON: str = "pipeline.json"

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
