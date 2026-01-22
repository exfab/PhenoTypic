"""Constants for CLI configuration.

This module centralizes all hardcoded configuration values used throughout
the CLI to enable easy tuning and maintenance.
"""

# SLURM configuration defaults
DEFAULT_SLURM_ARRAY_LIMIT = 1000
DEFAULT_SLURM_QUERY_TIMEOUT = 5  # seconds
SLURM_PROGRESS_POLL_INTERVAL = 10  # seconds

# Default image configuration
DEFAULT_GRID_ROWS = 8
DEFAULT_GRID_COLS = 12

# Event log configuration
MAX_TRACEBACK_LINES = 20
TRUNCATION_MARKER = "... (truncated) ..."

# Time validation (SLURM)
MIN_SLURM_TIME_MINUTES = 1
MAX_SLURM_TIME_MINUTES = 10080  # 7 days

# Job submission
SBATCH_SUBMISSION_TIMEOUT = 30  # seconds
